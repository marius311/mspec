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
               maps,
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
    pcls = load_pcls(pcls,maps=maps).sliced(lmax).rescaled(1e12)

    if (is_mpi_master()): print "Loading beams..."
    if isinstance(beams,load_files): beams = PowerSpectra(beams.load()).sliced(lmax)
    elif not isinstance(beams,PowerSpectra): raise ValueError("Beams should be load_files or PowerSpectra.")

    noise = defaultdict(lambda: 0)
    subpix = defaultdict(lambda: 0)
    calib = defaultdict(lambda: 1)

    if callable(weights): weights=weights(pcls)

    ps=set(chain(*[v.keys() for v in weights.values()]))
    ds=set(chain(*ps))


    # Load mode coupling matrices
    if get_covariance:
        if (is_mpi_master()): print "Loading kernels..."
        imlls, glls = load_kernels(kernels,lmax=lmax)
        tmasks = {((x,)+k[1:]):mask_name_transform(v) for k,v in masks.items() for x in (['T'] if k[0]=='T' else ['E','B'])}

        imlls = {(dm1,dm2):SymmetricTensorDict(imlls[tmasks[dm1],tmasks[dm2]],rank=4) for dm1,dm2 in ps}
        
        tgll_keys = ['TT','EE','BB','TE','EB','TB']

                        # TT     EE     BB     TE     EB     TB 
        tgll_entries = [['TTTT','TETE','TTTT','TTTT','TETE','TTTT'], #TT
                        ['TETE','EEEE','EEBB','EEEE','EEEE','EEBB'], #EE
                        ['TTTT','BBEE','EEEE','EEEE','EEEE','EEBB'], #BB
                        ['TTTT','EEEE','EEEE','TTTT','EEEE','TTTT'], #TE
                        ['TETE','EEEE','EEEE','EEEE','EBEB','EEEE'], #EB
                        ['TTTT','BBEE','BBEE','TTTT','EEEE','TTTT']] #TB
        tgll_dict = SymmetricTensorDict({(tuple(ki),tuple(kj)):(lambda x: (x[:2],x[2:]))(tgll_entries[i][j])
                                         for i,ki in enumerate(tgll_keys) 
                                         for j,kj in enumerate(tgll_keys)},rank=4)
        tglls = SymmetricTensorDict({((dm1,dm2),(dm3,dm4)):glls[(tmasks[dm1],tmasks[dm2]),(tmasks[dm3],tmasks[dm4])][tgll_dict[(dm1[0],dm2[0]),(dm3[0],dm4[0])]]
                                     for (dm1,dm2),(dm3,dm4) in pairs(pairs(ds))},rank=4)

    ells = arange(lmax)
    todl = ells*(ells+1)/2./pi

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
    
    if get_covariance:
        
        # The fiducial model for the mask deconvolved Cl's which gets used in the covariance
        fidcmb=SymmetricTensorDict(dict(zip([('T','T'),('E','E'),('B','B'),('T','E'),('E','B'),('T','B')],vstack([loadtxt(fidcmb)[:lmax].T,zeros((6,lmax))]))))
        fid_cls=get_fid_cls(pcls,beams,fidcmb)

        if (is_mpi_master()): print "Calculating Q terms..."
        Qterms = [((i,a),(j,b)) for a,b in ps for i,j in pairs('TEB')]
        def getQ(((i,a),(j,b))):
            imll = imlls.get((a,b),{}).get(((a[0],b[0]),(i,j)))
            if imll is not None: 
                return (((i,a),(j,b)), 
                        bin(todl*imll/beams[(a,b)],axis=1))
            else: 
                return None
        Q = dict([x for x in mpi_thread_map(getQ,Qterms) if x is not None])
        
        
        if (is_mpi_master()): print "Calculating detector covariance..."
        def pclcov((a,b),(c,d)):
            sym = lambda a,b: outer(a,b/2) + outer(b,a/2) #TODO: optimization here could improve speed
            return  (sym(fid_cls[(a,c)],fid_cls[(b,d)])*tglls[(a,c),(b,d)] if (a,c) in fid_cls and (b,d) in fid_cls else 0 + 
                     sym(fid_cls[(a,d)],fid_cls[(b,c)])*tglls[(a,d),(b,c)] if (a,d) in fid_cls and (b,c) in fid_cls else 0)

        def entry(((a,b),(c,d))):
            print "Calculating the %s entry."%(((a,b),(c,d)),)

            f = lambda i,a: (i,)+a[1:]

            return (((a,b),(c,d)),
                    array(sum(dot(Q[(i,a),(j,b)].T,dot(pclcov((f(i,a),f(j,b)),(f(k,c),f(l,d))),Q[(k,c),(l,d)]))
                        for (i,j) in pairs('TEB') for (k,l) in pairs('TEB')
                        if (((i,a),(j,b)) in Q and ((k,c),(l,d)) in Q)),dtype=float32))

        hat_cls_det.cov=SymmetricTensorDict(mpi_thread_map2(entry,pairs(ps),distribute=False),rank=4)


        # Equation (9)
        if (is_mpi_master()): 
            print "Calculating covariance..."
            for ((alpha,beta),(gamma,delta)) in pairs(sorted(weights.keys())):
                "Calculating the %s entry."%(((alpha,beta),(gamma,delta)),)
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

