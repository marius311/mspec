#!/usr/bin/env python
    
from mspec import *
from utils import *
import sys, os, re, gc
from numpy import *

#workaround when no-display
import matplotlib as mpl
mpl.rcParams['backend']='PDF'
import healpy as H

if __name__=="__main__":

    params = read_Mspec_ini(sys.argv[1:])
    
    def mt(m):
        return (m[0],'survey%s'%m[1])

    # Read map file names
    regex = re.compile(params.get("map_regex",def_map_regex))

    maps = [(os.path.join(params["maps"],f),regex.search(f)) for f in os.listdir(params["maps"]) if ".fits" in f]
    maps = sorted([(f,mt(r.groups())) for (f,r) in maps if r is not None])

    if (is_mpi_master()): print 'Found the following maps:\n '+"\n ".join(['%s (%s)'%(os.path.basename(f),id) for (f,id) in maps])
     
    # Other options
    weight1 = weight2 = params["mask"]
    
    # Get the alm's of each map
    def work(((map1,id1),(map2,id2))):
        clfile = os.path.join(params['pcls'],'%s__X__%s'%('-'.join(id1),'-'.join(id2)))
        print "Process %i is doing %s"%(get_mpi_rank(),clfile)
        ret = os.system("spice -mapfile %s -mapfile2 %s -weightfile %s -weightfile2 %s -pixelfile YES -subdipole YES -clfile %s &> /dev/null &"%(map1,map2,weight1,weight2,clfile))
        if ret!=0: raise ValueError("Spice crashed")

    mpi_map(work,pairs(maps))
