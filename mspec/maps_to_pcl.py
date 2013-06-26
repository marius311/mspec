#!/usr/bin/env python
    
from mspec import *
from utils import *
import sys, os, re
from numpy import *
from collections import defaultdict
import time

def maps_to_pcl(maps,
                mask,
                pcls,
                spice='spice',
                **kwargs):

    if isinstance(mask,str): 
        mask={mid:mask for mid in maps}

    if (is_mpi_master()): 
        print 'Found the following maps:\n '+"\n ".join(['%s - %s'%(id,os.path.basename(f)) for id,f in sorted(maps.items())])
     
    # Get the alm's of each map
    def work(((id1,map1),(id2,map2))):
        clfile = os.path.join(pcls,'%s__X__%s'%('-'.join(id1),'-'.join(id2)))
        weight1, weight2 = mask.get(id1), mask.get(id2)
        logfile = clfile+'.log'
        print "Process %i is doing %s"%(get_mpi_rank(),clfile)
        cmd = "%s -mapfile %s -mapfile2 %s -weightfile %s -weightfile2 %s -pixelfile YES -subdipole YES -clfile %s &> %s"%(spice,map1,map2,weight1,weight2,clfile,logfile)
        cmd += ' '*10000 #for some reason this solves an MPI crashing bug

        status = os.system(cmd)

        if status!=0: 
            if status==2: sys.exit()
            else:
                print "WARNING: Process %i crashed on %s with status %i"%(get_mpi_rank(),clfile,status)
                print cmd.trim()

    mpi_map(work,pairs(maps.items()))
