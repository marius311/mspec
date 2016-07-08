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
from multiprocessing import Pool as proc_Pool
from multiprocessing.dummy import Pool as thread_Pool
from functools import partial

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
               signal,
               kernels,
               masks=None,
               masks_forgll=None,
               detsignal=None,
               get_covariance=False,
               do_polarization=True,
               transform_tool='healpy',
               rescale=1e12,
               nl_eff=1,
               mask_name_transform=default_mask_name_transform,
               **kwargs):
    """
    """

    proc_pool = proc_Pool(get_num_threads())
    thread_pool = thread_Pool(get_num_threads())

    # Load stuff
    mspec_log("Loading pseudo-cl's...",rootlog=True)
    pcls = load_pcls(pcls,maps=maps,pool=proc_pool).sliced(lmax).rescaled(rescale)
    if not do_polarization:
        pcls = PowerSpectra({(a,b):v for (a,b),v in pcls.spectra.items() if a[0]==b[0]=='T'})


    mspec_log("Loading beams...",rootlog=True)
    if isinstance(beams,load_files): beams = PowerSpectra(beams.load(pool=proc_pool))
    elif not isinstance(beams,PowerSpectra): raise ValueError("Beams should be load_files or PowerSpectra.")
    beams = beams.sliced(lmax)

    noise = defaultdict(lambda: 0)
    subpix = defaultdict(lambda: 0)
    calib = defaultdict(lambda: 1)

    if callable(weights): weights=weights(pcls)

    ps=set(chain(*[v.keys() for v in weights.values()]))
    ds=set(chain(*ps))

    ells = arange(lmax)
    todl = ells*(ells+1)/2./pi
    pixwin = H.pixwin(2048)[:lmax]**2

    # Load the mode coupling matrices
    if get_covariance or transform_tool=='healpy':
        mspec_log("Loading kernels...",rootlog=True)

        if isinstance(masks,str):   masks={((x,)+mid):masks for mid in maps for x in 'TP'}
        if isinstance(masks_forgll,str): masks_forgll={((x,)+mid):masks_forgll for mid in maps for x in 'TP'}

        def get_tmasks(masks):
            return {((x,)+k[1:]):mask_name_transform(v) for k,v in masks.items() for x in (['T'] if k[0]=='T' else ['E','B'])}

        tmasks = get_tmasks(masks)
        if masks_forgll is not None: tmasks_forgll = get_tmasks(masks_forgll)
        else: tmasks_forgll = tmasks

        kerns = load_kernels(kernels)

        def glls((dm1,dm2),(dm3,dm4)):
            return kerns['gll'][(tmasks_forgll[dm1],tmasks_forgll[dm2]),(tmasks_forgll[dm3],tmasks_forgll[dm4])][(('T','T'),('T','T'))]

        def imlls(dm1,dm2):
            return kerns['imll'][tmasks[dm1],tmasks[dm2]]


    # Healpy doesn't deconvolve the mask & pixwin like spice, so do it by hand here
    if transform_tool=='healpy':
        mspec_log("Deconvolving masks...",rootlog=True)
        def deconv((a,b)):
            xs=[(x1,x2) for x1 in 'TEB' for x2 in 'TEB'
                if ((a[0],b[0]),(x1,x2)) in imlls(a,b)
                if (((x1,)+a[1:],(x2,)+b[1:]) in pcls)]
            if len(xs)>0: dpcls = sum(dot(imlls(a,b)[(a[0],b[0]),(x1,x2)],pcls[(x1,)+a[1:],(x2,)+b[1:]]) for x1,x2 in xs)
            else: dpcls = pcls[a,b]
            return ((a,b),dpcls/pixwin)
        pcls = PowerSpectra(dict(mpi_map2(deconv,pcls.get_spectra(),pool=thread_pool,distribute=True)))

    # The per detector signal estimate
    mspec_log("Calculating per-detector signal...",rootlog=True)
    hat_cls_det = PowerSpectra(ells=bin(ells),binning=bin)
    for (a,b) in ps:
        hat_cls_det[(a,b)] = bin(todl*(pcls[(a,b)] - (noise[a] if a==b else 0) - subpix[a,b])/beams[(a,b)])

    # The per frequency signal estimate
    mspec_log("Calculating signal...",rootlog=True)
    hat_cls_freq=PowerSpectra(ells=bin(ells),binning=bin)
    for fk in weights:
        hat_cls_freq[fk] = sum(hat_cls_det[mk]*w for mk,w in weights[fk].items())

    if get_covariance:

        fid_cls=get_fid_cls(pcls,beams,pixwin)

        mspec_log("Calculating Q terms...",rootlog=True)
        Qterms = [((i,a),(j,b)) for a,b in ps for i in 'TEB' for j in 'TEB']
        def getQ(((i,a),(j,b))):
            imll = imlls(a,b).get(((a[0],b[0]),(i,j)))
            if imll is not None:
                wl=beams[(a,b)].copy(); wl[wl<1e-2]=1e-2 #needed for a 100x100 numerical issue
                return (((i,a),(j,b)), bin(todl*imll/wl/pixwin,axis=0))
            else:
                return None
        Q = dict([x for x in mpi_map(getQ,Qterms,pool=thread_pool) if x is not None])

        mspec_log("Calculating detector covariance...",rootlog=True)
        def pclcov((a,b),(c,d)):
            sym = lambda a,b: outer(a,b/2) + outer(b,a/2) # This line about 1/3 of run time
            return  ((sym(fid_cls[(a,c)],fid_cls[(b,d)])*glls((a,c),(b,d)) if ((a,c) in fid_cls and (b,d) in fid_cls) else 0) +
                     (sym(fid_cls[(a,d)],fid_cls[(b,c)])*glls((a,d),(b,c)) if ((a,d) in fid_cls and (b,c) in fid_cls) else 0))

        def entry(((a,b),(c,d))):
            mspec_log("Calculating the %s entry."%(((a,b),(c,d)),))
            f = lambda i,a: (i,)+a[1:]
            return (((a,b),(c,d)),
                    array(sum(dot(Q[(i,a),(j,b)],dot(pclcov((f(i,a),f(j,b)),(f(k,c),f(l,d))),Q[(k,c),(l,d)].T))
                        for i in 'TEB' for j in 'TEB' for k in 'TEB' for l in 'TEB'
                        if (((i,a),(j,b)) in Q and ((k,c),(l,d)) in Q)),dtype=float32))

        hat_cls_det.cov=SymmetricTensorDict(mpi_map2(entry,pairs(ps),pool=thread_pool,distribute=False),rank=4)


        # Sum up the detector covariances to get the frequency one
        if (is_mpi_master()):
            mspec_log("Calculating covariance...",rootlog=True)
            for ((alpha,beta),(gamma,delta)) in pairs(sorted(weights.keys())):
                # "Calculating the %s entry."%(((alpha,beta),(gamma,delta)),)
                abcds=[((a,b),(c,d))
                       for (a,b) in weights[(alpha,beta)] for (c,d) in weights[(gamma,delta)]]
                hat_cls_freq.cov[((alpha,beta),(gamma,delta))] = \
                                 sum(weights[(alpha,beta)][(a,b)]*weights[(gamma,delta)][(c,d)]*hat_cls_det.cov[((a,b),(c,d))]
                                     for ((a,b),(c,d)) in abcds)

    if (is_mpi_master()):
        if detsignal is not None:
            try: os.makedirs(osp.dirname(detsignal))
            except: pass
            with open(detsignal,"w") as f: cPickle.dump(hat_cls_det,f,protocol=2)
        if signal is not None:
            try: os.makedirs(osp.dirname(signal))
            except: pass
            with open(signal,"w") as f: cPickle.dump(hat_cls_freq,f,protocol=2)
            with open(signal+'_meta',"w") as f: cPickle.dump({'pcls':pcls,
                                                              'beams':beams,
                                                              'fid_cls':fid_cls if get_covariance else None},f,protocol=2)


    return hat_cls_freq, hat_cls_det

