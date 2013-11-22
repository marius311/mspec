#!/usr/bin/env python
    
from mspec import *
from utils import *
import sys, os, re
from numpy import *
from collections import defaultdict
import time
import pyfits

def get_pcls(maps,
             masks,
             pcls,
             lmax,
             spice='spice',
             spicecache=None,
             do_polarization=False,
             overwrite=True,
             **kwargs):

    if isinstance(masks,str): 
        masks={mid:masks for mid in maps}

    if (is_mpi_master()): 
        print 'Found the following maps:\n '+"\n ".join(['%s - %s'%(id,os.path.basename(f)) for id,f in sorted(maps.items())])
        if not osp.exists(pcls): os.makedirs(pcls)

    def getcmd(((id1,map1),(id2,map2))):
        clfile = os.path.join(pcls,'%s__X__%s'%('-'.join(id1),'-'.join(id2)))
        weight1, weight2 = masks.get(('T',)+id1), masks.get(('T',)+id2)
        polweight1, polweight2 = masks.get(('P',)+id1,weight1), masks.get(('P',)+id2,weight2)
        logfile = clfile+'.log'
        pol = {True:'YES',False:'NO'}[do_polarization and haspol(map2)]
        if pol=='YES' and spicecache is not None:
            windowfile = check_spicecache(polweight1,polweight2,spicecache)[0]
        else:
            windowfile = None
            
        cmd_dict = {'mapfile':map1,
                    'mapfile2':map2,
                    'weightfile':weight1,
                    'weightfile2':weight2,
                    'polarization':pol,
                    'weightfilep':polweight1,
                    'weightfilep2':polweight2,
                    '_windowfile':windowfile,
                    'pixelfile':'YES',
                    'subdipole':'YES',
                    'nlmax':lmax,
                    'decouple':'YES',
                    'clfile':clfile}
                
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
        #cmd += ' &> /dev/null '
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
    jobs = zipwith(list({cmd['windowfile']:job for job,cmd in alljobs.items() if cmd.get('windowfile') is not None}.values()),False)
    jobs += zipwith(tjobs.keys(),False)
    jobs += zipwith(list(set(alljobs.keys()) - set([j for j,_ in jobs])),True)    
                
    if (is_mpi_master()): 
        print 'Doing the following %i jobs:\n '%len(jobs)+"\n ".join(['%s - %s'%((job[0][0],job[1][0]),alljobs[job]['clfile']) for job,_ in jobs])
    
    mpi_map2(dowork,[(alljobs[job],x) for job,x in jobs])
