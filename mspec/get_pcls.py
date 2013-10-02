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
             do_polarization=False,
             **kwargs):

    if isinstance(masks,str): 
        masks={mid:masks for mid in maps}

    if (is_mpi_master()): 
        print 'Found the following maps:\n '+"\n ".join(['%s - %s'%(id,os.path.basename(f)) for id,f in sorted(maps.items())])


    def dowork(((id1,map1),(id2,map2))):
        clfile = os.path.join(pcls,'%s__X__%s'%('-'.join(id1),'-'.join(id2)))
        weight1, weight2 = masks.get(('T',)+id1), masks.get(('T',)+id2)
        polweight1, polweight2 = masks.get(('P',)+id1,'YES'), masks.get(('P',)+id2,'YES')
        logfile = clfile+'.log'
        pol = {True:'YES',False:'NO'}[do_polarization and haspol(map2)]
        print "Process %i is doing %s"%(get_mpi_rank(),clfile)
        cmd_dict = {'mapfile':map1,
                    'mapfile2':map2,
                    'weightfile':weight1,
                    'weightfile2':weight2,
                    'polarization':pol,
                    'weightfilep':polweight1,
                    'weightfilep2':polweight2,
                    'pixelfile':'YES',
                    'subdipole':'YES',
                    'nlmax':lmax,
                    'decouple':'YES',
                    'clfile':clfile}
        cmd = ' '.join([spice]+['-%s %s'%(k,v) for k,v in cmd_dict.items()])
        cmd += ' '*10000 #for some reason this solves an MPI crashing bug
        print cmd.strip()

        status = os.system(cmd)

        if status!=0: 
            if status==2: sys.exit()
            else:
                print "WARNING: Process %i crashed on %s with status %i"%(get_mpi_rank(),clfile,status)
                print cmd.strip()
    

    polmaps = {id1:map1 for (id1,map1) in maps.items() 
               if do_polarization and 'Q_Stokes' in pyfits.open(map1)[1].header.values()}
    tempmaps = dict(set(maps.items()) - set(polmaps.items()))

    work  = pairs(tempmaps.items())
    work += [(m1,m2) for m1 in tempmaps.items() for m2 in polmaps.items()]
    work += [(m1,m2) for m1 in polmaps.items()  for m2 in polmaps.items()]

    mpi_map(dowork,work)
