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



def pcl_to_signal(lmax,
                  bin,
                  pcls,
                  beams,
                  weights,
                  fidcmb,
                  signal,
                  mask=None,
                  detsignal=None,
                  get_covariance=False,
                  **kwargs):
    """
    """
  

    # Load stuff
    if (is_mpi_master()): print "Loading pseudo-cl's, beams, and noise..."
    pcls = load_pcls(pcls).sliced(lmax).rescaled(1e12)
    beams = PowerSpectra(beams.load()).sliced(lmax)
    noise = defaultdict(lambda: 0)
    subpix = defaultdict(lambda: 0)
    calib = defaultdict(lambda: 1)

    if callable(weights): weights=weights(pcls)

    # Load mode coupling matrices
    if mask is not None:
        if (is_mpi_master()): print "Loading mode coupling matrices..."
        imll = load_multi(mask+".imll")
        if get_covariance: gll2 = load_multi(mask+".mll2")/(2*arange(imll.shape[0])+1)
        else: gll2 = None
    else:
        imll=1
        gll2=diag(1./(2*arange(lmax)+1))

    # Check lmax
    #lmax_union = min([imll.shape[0],min(min(b.shape[0] for b in ps.spectra.values()) for ps in [beam,pcls,noise,subpix] if isinstance(ps,PowerSpectra))])
    #if lmax_union < lmax and is_mpi_master(): print "Warning: Lowering lmax from %i to %i because of insufficient data."%(lmax,lmax_union)
    #for ps in ['beam','pcls','noise','subpix']:
    #    exec("if isinstance(%s,PowerSpectra): %s = %s.sliced(0,lmax)"%((ps,)*3))
    imll = imll[:lmax,:lmax]
    if gll2 is not None: gll2 = gll2[:lmax,:lmax]
    ells = arange(lmax)
    todl = ells*(ells+1)/2/pi

    #Equation (4), the per detector signal estimate
    if (is_mpi_master()): print "Calculating per-detector signal..."
    hat_cls_det = PowerSpectra(ells=bin(ells))
    for (a,b) in pcls.spectra: 
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
            pclcov = (lambda x: x+x.T)(outer(fid_cls[(a,c)],fid_cls[(b,d)]) + outer(fid_cls[(a,d)],fid_cls[(b,c)]))*gll2/2
            # Equation (7)
            entry = dot(bin(transpose([todl])*(imll/transpose([beams[(a,b)]])),axis=0),dot(pclcov,bin(transpose([todl])*(imll/transpose([beams[(c,d)]])),axis=0).T))*calib[a]*calib[b]*calib[c]*calib[d]
            return (((a,b),(c,d)),entry)

        maps=list(chain(*[v.keys() for v in weights.values()]))
        hat_cls_det.cov=SymmetricTensorDict(mpi_map(entry,pairs(maps)),rank=4)
        
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

