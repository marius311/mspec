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
               do_polarization=True,
               transform_tool='healpy',
               rescale=1e12,
               nl_eff=1,
               mask_name_transform=default_mask_name_transform,
               **kwargs):
    """
    """


    # Load stuff
    if (is_mpi_master()): print "Loading pseudo-cl's..."
    pcls = load_pcls(pcls,maps=maps).sliced(lmax).rescaled(rescale)

    if (is_mpi_master()): print "Loading beams..."
    if isinstance(beams,load_files): beams = PowerSpectra(beams.load()).sliced(lmax)
    elif not isinstance(beams,PowerSpectra): raise ValueError("Beams should be load_files or PowerSpectra.")

    noise = defaultdict(lambda: 0)
    subpix = defaultdict(lambda: 0)
    calib = defaultdict(lambda: 1)

    if callable(weights): weights=weights(pcls)

    ps=set(chain(*[v.keys() for v in weights.values()]))
    ds=set(chain(*ps))

    ells = arange(lmax)
    todl = ells*(ells+1)/2./pi
    pixwin = H.pixwin(2048)[:lmax]**2

    # Load mode coupling matrices
    if get_covariance or transform_tool=='healpy':
        if (is_mpi_master()): print "Loading kernels..."
        if masks:

            if isinstance(masks,str):
                masks={((x,)+mid):masks for mid in maps for x in 'TP'}

            imlls, glls = load_kernels(kernels,lmax=lmax,pol=do_polarization)
            tmasks = {((x,)+k[1:]):mask_name_transform(v) for k,v in masks.items() for x in (['T'] if k[0]=='T' else ['E','B'])}

            imlls = {(dm1,dm2):SymmetricTensorDict(imlls[tmasks[dm1],tmasks[dm2]],rank=4) for dm1,dm2 in pcls.spectra}

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
        else:
            imll1 = {((x1,x2),(x1,x2)):identity(lmax) for x1,x2 in ['TT','EE','BB','TE','EB','TB']}
            imlls = {(dm1,dm2):imll1 for dm1,dm2 in pcls.get_spectra()}


        def tglls((dm1,dm2),(dm3,dm4)):
            if masks:
                return glls[(tmasks[dm1],tmasks[dm2]),(tmasks[dm3],tmasks[dm4])][tgll_dict[(dm1[0],dm2[0]),(dm3[0],dm4[0])]]
            else:
                return diag(1./(2*ells+1))


    #healpy doesn't deconvolve the mask/pixwin like spice, so do it by hand here
    if transform_tool=='healpy':
        for (a,b) in pcls.get_spectra():
            xs=[(x1,x2) for x1 in 'TEB' for x2 in 'TEB'
                if ((a[0],b[0]),(x1,x2)) in imlls[a,b]
                if (((x1,)+a[1:],(x2,)+b[1:]) in pcls) and
                (((a[0],b[0]),(x1,x2)) not in chain(*[((x,x[::-1]),(x[::-1],x))
                                                      for x in [('T','E'),('T','B'),('E','B')]]))] #TODO: combine this with Q term symmetry thing below
            if len(xs)>0: pcls[(a,b)] = sum(dot(imlls[a,b][(a[0],b[0]),(x1,x2)],pcls[(x1,)+a[1:],(x2,)+b[1:]]) for x1,x2 in xs)
            pcls[(a,b)] = pcls[(a,b)]/pixwin

    #Equation (4), the per detector signal estimate
    if (is_mpi_master()): print "Calculating per-detector signal..."
    hat_cls_det = PowerSpectra(ells=bin(ells),binning=bin)
    for (a,b) in ps:
        hat_cls_det[(a,b)] = bin(todl*(pcls[(a,b)] - (noise[a] if a==b else 0) - subpix[a,b])/beams[(a,b)])

    # Equation (6), the per frequency signal estimate
    if (is_mpi_master()): print "Calculating signal..."
    hat_cls_freq=PowerSpectra(ells=bin(ells),binning=bin) #TODO binning
    for fk in weights:
        hat_cls_freq[fk] = sum(hat_cls_det[mk]*w for mk,w in weights[fk].items())

    if get_covariance:

        # The fiducial model for the mask deconvolved Cl's which gets used in the covariance
        fidcmb=SymmetricTensorDict(dict(zip([('T','T'),('E','E'),('B','B'),('T','E'),('E','B'),('T','B')],vstack([loadtxt(fidcmb)[:,:lmax],zeros((6,lmax))]))))
        fid_cls=get_fid_cls(pcls,beams,fidcmb,pixwin,nl_eff=nl_eff)

        if (is_mpi_master()): print "Calculating Q terms..."
        Qterms = [((i,a),(j,b)) for a,b in ps for i in 'TEB' for j in 'TEB']
        def getQ(((i,a),(j,b))):
            imll = imlls.get((a,b),{}).get(((a[0],b[0]),(i,j)))
            if (imll is not None and
                #Because Q symmetry is not same as what I assume for imlls (which are a SymmetricTensorDict(rank=4)):
                ((a[0],b[0]),(i,j)) not in chain(*[((x,x[::-1]),(x[::-1],x))
                                                   for x in [('T','E'),('T','B'),('E','B')]])):
                wl=beams[(a,b)].copy(); wl[wl<1e-2]=1e-2 #needed for a 100x100 numerical issue
                return (((i,a),(j,b)), bin(todl*imll/wl/pixwin,axis=0))
            else:
                return None
        Q = dict([x for x in (mpi_thread_map if (get_mpi_size()!=1) else map)(getQ,Qterms) if x is not None])


        if (is_mpi_master()): print "Calculating detector covariance..."
        def pclcov((a,b),(c,d)):
            sym = lambda a,b: outer(a,b/2) + outer(b,a/2) #TODO: optimization here could improve speed
            return  ((sym(fid_cls[(a,c)],fid_cls[(b,d)])*tglls((a,c),(b,d)) if ((a,c) in fid_cls and (b,d) in fid_cls) else 0) +
                     (sym(fid_cls[(a,d)],fid_cls[(b,c)])*tglls((a,d),(b,c)) if ((a,d) in fid_cls and (b,c) in fid_cls) else 0))

        def entry(((a,b),(c,d))):
            print "Calculating the %s entry."%(((a,b),(c,d)),)

            f = lambda i,a: (i,)+a[1:]

            return (((a,b),(c,d)),
                    array(sum(dot(Q[(i,a),(j,b)],dot(pclcov((f(i,a),f(j,b)),(f(k,c),f(l,d))),Q[(k,c),(l,d)].T))
                        for i in 'TEB' for j in 'TEB' for k in 'TEB' for l in 'TEB'
                        if (((i,a),(j,b)) in Q and ((k,c),(l,d)) in Q)),dtype=float32))

        # return locals()

        hat_cls_det.cov=SymmetricTensorDict((mpi_thread_map2 if (get_mpi_size()!=1) else lambda *a, **_: map(*a))(entry,pairs(ps),distribute=False),rank=4)

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
            with open(detsignal,"w") as f: cPickle.dump(hat_cls_det,f,protocol=2)
        if signal is not None:
            with open(signal,"w") as f: cPickle.dump(hat_cls_freq,f,protocol=2)

    return hat_cls_freq, hat_cls_det

