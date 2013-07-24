#!/usr/bin/env python

from mspec import *
import sys, os, re
from numpy import *
from numpy.linalg import norm
from collections import namedtuple, defaultdict
from utils import *
import utils
from bisect import bisect_right
from itertools import chain
import cPickle

#workaround when no-display
#import matplotlib as mpl
#mpl.rcParams['backend']='PDF'
import healpy as H



def get_signal(lmax,
               bin,
               pcls,
               beams,
               weights,
               fidcmb,
               signal,
               kernels,
               masks=None,
               detsignal=None,
               get_covariance=False,
               mask_name_transform=default_mask_name_transform,
               **kwargs):
    """
    """
  

    # Load stuff
    if (is_mpi_master()): print "Loading pseudo-cl's..."
    pcls = load_pcls(pcls).sliced(lmax).rescaled(1e12)

    if (is_mpi_master()): print "Loading beams..."
    beams = PowerSpectra(beams.load()).sliced(lmax)

    noise = defaultdict(lambda: 0)
    subpix = defaultdict(lambda: 0)
    calib = defaultdict(lambda: 1)

    if callable(weights): weights=weights(pcls)

    ps=list(chain(*[v.keys() for v in weights.values()]))

    # Load mode coupling matrices
    if get_covariance:
        if (is_mpi_master()): print "Loading kernels..."
        imlls, glls = load_kernels(kernels,lmax=lmax)
        tmasks = {k:mask_name_transform(v) for k,v in masks.items()}        
        imlls = SymmetricTensorDict({(dm1,dm2):imlls[tmasks[dm1],tmasks[dm2]] 
                                     for dm1,dm2 in ps})
        glls = SymmetricTensorDict({((dm1,dm2),(dm3,dm4)):glls[(tmasks[dm1],tmasks[dm2]),(tmasks[dm3],tmasks[dm4])] 
                                    for (dm1,dm2),(dm3,dm4) in pairs(ps)})

    
    ells = arange(lmax)
    todl = ells*(ells+1.)/2/pi

    #Equation (4), the per detector signal estimate
    if (is_mpi_master()): print "Calculating per-detector signal..."
    hat_cls_det = PowerSpectra(ells=bin(ells))
    for (a,b) in ps: 
        hat_cls_det[(a,b)] = bin(todl*(pcls[(a,b)] - (noise[a] if a==b else 0) - subpix[a,b])/beams[(a,b)])

    # Equation (6), the per frequency signal estimate
    if (is_mpi_master()): print "Calculating signal..."
    hat_cls_freq=PowerSpectra(ells=bin(ells)) #TODO binning
    for fk in weights:
        hat_cls_freq[fk] = sum(hat_cls_det[mk]*w for mk,w in weights[fk].items())
    
    # The fiducial model for the mask deconvolved Cl's which gets used in the covariance
    if get_covariance:
        if (is_mpi_master()): print "Calculating fiducial signal..."

        fidcmb=loadtxt(fidcmb)[:lmax]
        fidcmb[1:]/=(arange(1,lmax)*(arange(1,lmax)+1)/2/pi)

        def fidize(x,(tl,tu)=(200,500),window_len=200):
            x=smooth(x,window_len=window_len)
            x[tl:tu]*=sin(pi*arange(tu-tl)/(tu-tl)/2)**2
            x[:tl]=0
            return x

        fid_cls=SymmetricTensorDict()

        for k in pcls.spectra:
            bcmb = fidcmb*beams[k]
            fid_cls[k]=bcmb + fidize(pcls[k]-bcmb)


        if (is_mpi_master()): print "Calculating detector covariance..."
        
        def entry(((a,b),(c,d))):
            print "Calculating the %s entry."%(((a,b),(c,d)),)
            # Equation (5)
            pclcov = (lambda x: x+x.T)(outer(fid_cls[(a,c)],fid_cls[(b,d)]) + outer(fid_cls[(a,d)],fid_cls[(b,c)]))*glls[((a,b),(c,d))]/2
            # Equation (7)
            entry = dot(bin(transpose([todl])*(imlls[(a,b)]/transpose([beams[(a,b)]])),axis=0),dot(pclcov,bin(transpose([todl])*(imlls[(c,d)]/transpose([beams[(c,d)]])),axis=0).T))*calib[a]*calib[b]*calib[c]*calib[d]
            return (((a,b),(c,d)),entry)

        hat_cls_det.cov=SymmetricTensorDict(mpi_thread_map(entry,pairs(ps)),rank=4)
        
        # Equation (9)
        if (is_mpi_master()): 
            print "Calculating covariance..."
            for ((alpha,beta),(gamma,delta)) in pairs(sorted(weights.keys())):
                print ((alpha,beta),(gamma,delta))
                abcds=[((a,b),(c,d)) 
                       for (a,b) in weights[(alpha,beta)] for (c,d) in weights[(gamma,delta)]]
                hat_cls_freq.cov[((alpha,beta),(gamma,delta))] = \
                                  sum(weights[(alpha,beta)][(a,b)]*weights[(gamma,delta)][(c,d)]*hat_cls_det.cov[((a,b),(c,d))] 
                                      for ((a,b),(c,d)) in abcds)


    if (is_mpi_master()): 
        if detsignal is not None: 
            with open(detsignal,"w") as f: cPickle.dump(hat_cls_det,f)
        with open(signal,"w") as f: cPickle.dump(hat_cls_freq,f)

    return hat_cls_freq

