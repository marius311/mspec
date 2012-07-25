#!/usr/bin/env python
    
from mspec import *
from utils import *
import sys, os, re, gc
from numpy import *
import healpy as H

if __name__=="__main__":

    params = read_Mspec_ini(sys.argv[1:])
    
    # Read map file names
    regex = re.compile(params.get("map_regex",def_map_regex))
    maps = [(os.path.join(params["maps"],f),regex.search(f)) for f in os.listdir(params["maps"]) if ".fits" in f]
    maps = sorted([(f,[r.group(1),r.group(2)]) for (f,r) in maps if r and ('freqs' not in params or r.group(1) in params['freqs'])])
    if (is_mpi_master()): print 'Found the following maps:\n '+"\n ".join([os.path.basename(f)+' ('+str(MapID(fr,'T',id))+')' for (f,(fr,id)) in maps])
     
    # Other options
    mask = H.read_map(params["mask"]) if params.get("mask") else None
    lmax = params["lmax"]
    ells = arange(lmax)
    
    # Get nside from the first map and make sure the mask is the same
    # params["nside"]=H.npix2nside(pyfits.open(maps[0][0])[1].data.shape[0])*32
    params["nside"]=2048
    if (mask!=None and H.npix2nside(alen(mask))!=params["nside"]): mask = H.ud_grade(mask,params["nside"])

    # Get the alm's of each map
    def maps2alm((file,det)):
        print "Process "+str(get_mpi_rank())+" is transforming '"+file+"'"
        mp = H.read_map(file)
        if (len(mp[mp<-1e20])!=0):
            print "Warning: Setting "+str(len(mp[mp<-1e20]))+" remaining UNSEEN pixels after masking to 0 in "+file
            mp[mp<-1e20]=0
        if mask!=None: mp*=mask
        det.insert(1,"T")
        return (file,det,H.map2alm(mp,lmax=lmax))
        
    alms = mpi_map(maps2alm,maps,distribute=True)    
    
    # Figure out all the powerspectra we need to compute and their output file names
    almpairs = [(alm1,alm2,os.path.join(params["pcls"],"-".join(d1)+'__'+"-".join(d2))) for ((m1,d1,alm1),(m2,d2,alm2)) in pairs(alms)]
    
    def almpair2ps((alm1,alm2,output)):
        print "Process "+str(get_mpi_rank())+" is calculating '"+output+"'"
        save_multi(output,alm2cl(alm1,alm2)*1e12,npy=False)

    mpi_map(almpair2ps,almpairs)
