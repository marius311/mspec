#!/usr/bin/env python

from mspec import *
import healpy as H
import sys, os, re
from numpy import *
from mll import mll
from utils import *
from numpy.linalg import inv

def mask_to_mll(params):
    #Read in parameter file and get output names
    print "Getting mode coupling for '"+params.mask+"'"
    
    #Create the healpix mask
    mask = H.read_map(params.mask)
    
    #Read in other parameters
    lmax = int(params.lmax)+2
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
        save_multi(params.mask+"."+post,mat)
        print "Saved '"+params.mask+"."+post+"'"
