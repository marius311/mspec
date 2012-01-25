#!/usr/bin/env python
    
"""
 This script computes the all possible auto and cross spectra from a list
 of fits maps, and outputs one powerspectrum per file with a descriptive 
 file name.
 
 Currently only TT is supported.
 

 Usage: 
     python maps_to_pcl.py parameter_file.ini
     
     OR 
     
     mpiexec -n 8 python maps_to_pcl.py parameter_file.ini
 
 Paramter file:
 
     maps - Folder where the map fits files are
     lmax - Maximum ell when calculating the powerspectrum
     powerspectra - A folder where to write the output files.
     
     Optional-
     
     mask - A mask to apply before calculating the powerspectrum, default None
     map_freq_regex/map_det_regex - Regular Expression that when applied to the map filename, the 
                                    first group returns the frequency/detector id. Defaults to 
                                    accepting names like '100-1a_W_TauDeconv_v47.fits'
     
"""

from americanpypeline import *
from utils import *
import sys, os, re, gc, pyfits
from numpy import *
import healpy as H

if __name__=="__main__":

    if (len(sys.argv) != 2): 
        print "Usage: python maps_to_pcl.py parameter_file.ini"
        sys.exit()
       
    params = read_AP_ini(sys.argv[1])
    
    # Read map file names
    maps = [os.path.join(params["maps"],f) for f in os.listdir(params["maps"]) if f.rfind(".fits")!=-1]
    
    # Get frequency/detector id from each map file name
    ids = [re.compile(params.get(k,d)) for (k,d) in [("map_freq_regex","(100|143|217|353)"),("map_det_regex","-([1-8][abc]?)_")]]
    maps = sorted([(m,[i.group(1) for i in id]) for (m,id) in [(m,[id.search(m) for id in ids]) for m in maps] if None not in id])
    
    # Other options
    mask = H.read_map(params["mask"])
    fsky = sum(mask)/alen(mask)
    lmax = params["lmax"]
    ells = arange(lmax)
    
    # Get nside from the first map and make sure the mask is the same
    params["nside"]=H.npix2nside(pyfits.open(maps[0][0])[1].data.shape[0])*32
    if (H.npix2nside(alen(mask))!=params["nside"]): mask = H.ud_grade(mask,params["nside"])

    # Get the alm's of each map
    def maps2alm((file,det)):
        print "Process "+str(get_mpi_rank())+" is transforming '"+file+"'"
        mp = H.read_map(file)
        if mask!=None: mp*=mask
        det.insert(1,"T") #For now just T maps, in future we'll have either separate files or 3 columns, handled here
        return (file,det,H.map2alm(mp,lmax=lmax))
        
    alms = mpi_map(maps2alm,maps,distribute=True)    
    
    # Figure out all the powerspectra we need to compute and their output file names
    almpairs = [(alm1,alm2,os.path.join(params["pcls"],"-".join(d1)+'__'+"-".join(d2)+'.dat')) for ((m1,d1,alm1),(m2,d2,alm2)) in pairs(alms)]
    
    def almpair2ps((alm1,alm2,output)):
        print "Process "+str(get_mpi_rank())+" is calculating '"+output+"'"
        savetxt(output,alm2cl(alm1,alm2)*1e12)#/H.pixwin(params["nside"])[:lmax]**2)

    mpi_map(almpair2ps,almpairs)
