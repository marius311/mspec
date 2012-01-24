#!/usr/bin/env python


"""
Given a mask, this script computes the mode coupling matrix and its inverse for both the mask
and for the mask squared (needed for covariance).

Usage:
    python mask_to_mll.py param_file.ini
    
    where param_file has a line which can look like:
        mask = ptsrc.fits
        OR
        mask = 70% + ptsrc.fits
    
    corresponding to a point source mask or a point source mask + 70% sky galactic cut respectively.
    
Output:
    Four files, one for each matrix, are created using the specified fits file as a root file name. 
    
"""

from americanpypeline import *
import healpy as H
import sys, os, re
from numpy import *
from mll import mll
from utils import *
from numpy.linalg import inv


if __name__=="__main__":
    if (len(sys.argv) != 2): 
        print "Usage: python mask_to_mll.py parameter_file.ini"
        sys.exit()
        
    #Read in parameter file and get output names
    params = read_AP_ini(sys.argv[1])
    print "Getting mode coupling for '"+params["mask"]+"'"
    
    #Create the healpix mask
    mask = H.read_map(params["mask"])
    
    #Read in other parameters
    lmax = int(params["lmax"])
    wlmax = min(2*lmax+1,3*H.npix2nside(alen(mask))-1)
    
    print "Getting mask powerspectrum..."
    wl = H.anafast(mask,lmax=wlmax)
    wl2 = H.anafast(mask**2,lmax=wlmax)
    if (alen(wl)<2*lmax):
        print "Warning: Your mask NSIDE isn't high enough calculate the mask powerspectrum to \
                the necessary ell=2*lmax. Padding with zeros."
        wl=append(wl,zeros(2*lmax-alen(wl)+1))
        wl2=append(wl,zeros(2*lmax-alen(wl2)+1))
        
    print "Getting mode coupling matrix..."
    m = mll.getmll(wl,lmax)
    m2 = mll.getmll(wl2,lmax)
    im=inv(m)
    im2=inv(m2)
    
    #Save results
    for (post,mat) in [("mll",m),("imll",im),("mll2",m2),("imll2",im2)]: 
        savetxt(params["mask"]+"."+post,mat)
        print "Saved '"+params["mask"]+"."+post+"'"
