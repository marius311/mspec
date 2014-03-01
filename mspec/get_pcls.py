#!/usr/bin/env python
    
from mspec import *
from utils import *
import sys, os, re
from numpy import *
from collections import defaultdict
import time
import pyfits
import healpy as H


def get_pcls(transform_tool='healpy',
             **kwargs):
    if transform_tool=='spice': 
        return get_pcls_spice(**kwargs)
    elif transform_tool=='healpy': 
        return get_pcls_healpy(**kwargs)
    else:
        raise ValueError("transform_tool should be on of ['spice','healpy']")
    
def get_pcls_spice(maps,
                   masks,
                   pcls,
                   lmax,
                   spice='spice',
                   spicecache=None,
                   do_polarization=False,
                   overwrite=True,
                   spiceargs=None,
                   **kwargs):

    if isinstance(masks,str): 
        masks={((x,)+mid):masks for mid in maps for x in 'TP'}

    if (is_mpi_master()): 
        print 'Found the following maps:\n '+"\n ".join(['%s - %s'%(id,os.path.basename(f)) for id,f in sorted(maps.items())])
        if not osp.exists(pcls): os.makedirs(pcls)

    def getcmd(((id1,map1),(id2,map2))):
        clfile = os.path.join(pcls,'%s__X__%s'%('-'.join(id1),'-'.join(id2)))
        weight1, weight2 = masks.get(('T',)+id1), masks.get(('T',)+id2)
        polweight1, polweight2 = masks.get(('P',)+id1,weight1), masks.get(('P',)+id2,weight2)
        logfile = clfile+'.log'
        pol = {True:'YES',False:'NO'}[do_polarization and haspol(map2)]
        if (pol=='YES' and spicecache is not None): 
            windowfile = check_spicecache(polweight1,polweight2,spicecache)[0]
        else:
            windowfile = None
            
        cmd_dict = {'mapfile':map1,
                    'mapfile2':map2,
                    'polarization':pol,
                    '_windowfile':windowfile,
                    'pixelfile':'YES',
                    'subdipole':'YES',
                    'nlmax':lmax,
                    'decouple':'YES',
                    'clfile':clfile}
        if weight1 is not None: cmd_dict['weightfile']=weight1
        if weight2 is not None: cmd_dict['weightfile2']=weight2
        if polweight1 is not None: cmd_dict['weightfile']=polweight1
        if polweight2 is not None: cmd_dict['weightfilep2']=polweight2
        if spiceargs is not None: cmd_dict.update(spiceargs)
                
        return cmd_dict
    
    def dowork((cmd_dict,waitforwindow)):
    
        windowfile = cmd_dict.pop('_windowfile',None)
        if windowfile is not None:
            if osp.exists(windowfile):
                cmd_dict['windowfilein']=windowfile
            elif waitforwindow:
                cmd_dict['windowfilein']=windowfile
                print 'Process %i is waiting for cached window %s for file %s...'%(get_mpi_rank(),windowfile,cmd_dict['clfile'])
                while not osp.exists(windowfile): time.sleep(1)
            else: cmd_dict['windowfileout']=windowfile

        print "Process %i is doing %s"%(get_mpi_rank(),cmd_dict['clfile'])
        if 'windowfileout' in cmd_dict: print ' (creating cache %s)'%cmd_dict['windowfileout']
        elif 'windowfilein' in cmd_dict: print ' (using cache %s)'%cmd_dict['windowfilein']
        
        cmd = ' '.join([spice]+['-%s %s'%(k,v) for k,v in cmd_dict.items()]) 
        cmd += ' &> /dev/null '
        cmd += ' '*10000 #for some reason this solves an MPI crashing bug

        start=time.time()
        status = os.system(cmd)

        if status!=0: 
            if status==2: sys.exit()
            else:
                print "WARNING: Process %i crashed on %s with status %i"%(get_mpi_rank(),cmd_dict['clfile'],status)
                print cmd.strip()
        print 'Process %i finished %s in %.1f seconds.'%(get_mpi_rank(),cmd_dict['clfile'],time.time()-start)


    pmaps = {id1:map1 for (id1,map1) in maps.items() 
             if do_polarization and haspol(map1)}
    tmaps = dict(set(maps.items()) - set(pmaps.items()))
    tjobs = {j:getcmd(j) for j in pairs(tmaps.items())}
    pjobs = {j:getcmd(j) for j in ([(m1,m2) for m1 in pmaps.items() for m2 in pmaps.items()] +
                                   [(m1,m2) for m1 in tmaps.items() for m2 in pmaps.items()])}
             
    if not overwrite:
        #remove jobs which are already done
        existing = []
        for jobs in [tjobs,pjobs]:
            for job,cmd in jobs.items(): 
                if osp.exists(cmd['clfile']):
                    jobs.pop(job)
                    existing.append((job,cmd['clfile']))
        if (is_mpi_master()): 
            print 'The following %i jobs are already done:\n '%len(existing)+"\n ".join(['%s - %s'%((id1,id2),clfile) for ((id1,_),(id2,_)),clfile in sorted(existing)])
                    
    #order the jobs so precomputation of windowfile happens efficiently
    alljobs = {}; alljobs.update(tjobs); alljobs.update(pjobs);
    def zipwith(x,val): return zip(x,[val]*len(x))
    jobs = zipwith(list({cmd['_windowfile']:job for job,cmd in alljobs.items() if cmd.get('_windowfile') is not None}.values()),False)
    jobs += zipwith(tjobs.keys(),False)
    jobs += zipwith(list(set(alljobs.keys()) - set([j for j,_ in jobs])),True)    
                
    if (is_mpi_master()): 
        print 'Doing the following %i jobs:\n '%len(jobs)+"\n ".join(['%s - %s'%((job[0][0],job[1][0]),alljobs[job]['clfile']) for job,_ in jobs])
    
    mpi_map2(dowork,[(alljobs[job],x) for job,x in jobs])
    
    
def get_pcls_healpy(maps,
                    masks,
                    pcls,
                    lmax,
                    do_polarization=False,
                    **kwargs):
    
    if isinstance(masks,str): 
        masks={mid:masks for mid in maps}

    if (is_mpi_master()): 
        print 'Found the following maps:\n '+"\n ".join(['%s - %s'%(id,os.path.basename(f)) for id,f in sorted(maps.items())])
        if not osp.exists(pcls): os.makedirs(pcls)

    def getalm((id,m)):
        print "Process %i: Getting alm's for %s - %s"%(get_mpi_rank(),id,m)
        m=array(H.read_map(m,field=[0] if id in tmaps else [0,1,2]))
        tmask=H.read_map(masks.get(('T',)+id)) if masks.get(('T',)+id) is not None else 1
        if id in pmaps: 
            pmask=H.read_map(masks.get(('P',)+id)) if masks.get(('P',)+id) is not None else 1
            m[0]*=tmask; m[1]*=pmask; m[2]*=pmask
        else:
            m*=tmask
            
        badidx=m<-1e20
        nbad=badidx.sum()
        if nbad>0:
            print "Warning setting %i (%.3f%%) UNSEEN pixels to zero."%(nbad,100*nbad/(12*2048**2.))
            m[badidx]=0
         
        return (id,array(atleast_2d(H.map2alm(m,lmax=lmax)),dtype=float32))
        
    pmaps = {id1:map1 for (id1,map1) in maps.items() 
             if do_polarization and haspol(map1)}
    tmaps = dict(set(maps.items()) - set(pmaps.items()))
    
    alms = dict(mpi_map(getalm,pmaps.items()+tmaps.items()))
    
    def writecl((id1,id2)):
        clfile = os.path.join(pcls,'%s__X__%s'%('-'.join(id1),'-'.join(id2)))
        print 'Process %i: Getting cls for (%s,%s) - %s'%(get_mpi_rank(),id1,id2,clfile)
        alm1=alms[id1]; alm2=alms[id2]
        spicexs=['TT','EE','BB','TE','TB','EB']
        cls={(x1+x2):H.alm2cl(a1,a2)
             for x1,a1 in zip('TEB',alm1) 
             for x2,a2 in zip('TEB',alm2)
             if (x1+x2) in spicexs}
        savetxt(clfile,vstack([arange(lmax+1)]+[cls.get(x,zeros(lmax+1)) for x in spicexs]).T,fmt='%i '+' '.join(['%.4g']*6))
    
    tjobs = pairs(tmaps)
    pjobs = ([(m1,m2) for m1 in pmaps for m2 in pmaps] +
             [(m1,m2) for m1 in tmaps for m2 in pmaps])
    alljobs = tjobs+pjobs
             
    mpi_thread_map(writecl,alljobs)
