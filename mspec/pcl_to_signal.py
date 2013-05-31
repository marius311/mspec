#!/usr/bin/env python

from mspec import *
import sys, os, re
from numpy import *
from numpy.linalg import norm
from collections import namedtuple, defaultdict
from utils import *
import utils
from bisect import bisect_right

#workaround when no-display
import matplotlib as mpl
mpl.rcParams['backend']='PDF'
import healpy as H



if __name__=="__main__": 
    
    params = read_Mspec_ini(sys.argv[1:])
    lmax = params["lmax"]
    bin = params["binning"]

    
    # Load stuff
    if (is_mpi_master()): print "Loading pseudo-cl's, beams, and noise..."
    pcls = load_pcls(params)
    beam = load_beams(params) 
    noise = load_noise(params) if params.get("noise",None) else defaultdict(lambda: 0)
    subpix = load_subpix(params) if params.get("subpix",None) else defaultdict(lambda: 0)
    
    freqs = params.get("freqs")
    if 'maps_ids' in params:
        maps = set(MapID(m) for m in params.get('maps_ids'))
    else:
        maps = set(m for m in pcls.get_maps() if (m.fr in freqs if freqs!=None else True))
    if params.get("beams",None): maps&=set(beam.get_maps())
    if params.get("noise",None): maps&=set(noise.get_keys())
    freqs = set(m.fr for m in maps)
    
    if (is_mpi_master()): print "Found pseudo-cl's, beams, and noise for: "+str(sorted(maps))

    if str2bool(params.get("use_auto_spectra",'F')): weight = defaultdict(lambda: 1)
    else: 
        if str2bool(params.get("optimal_weights",'T')): weight = get_optimal_weights(pcls)
        else: weight = {(a,b): 0 if a==b else 1 for a in pcls.get_maps() for b in pcls.get_maps()}


    # Load mode coupling matrices
    if (params.get("mask")):
        if (is_mpi_master()): print "Loading mode coupling matrices..."
        imll = load_multi(params["mask"]+".imll")
        if str2bool(params.get("get_covariance",False)): gll2 = load_multi(params["mask"]+".mll2")/(2*arange(imll.shape[0])+1)
        else: gll2 = None
    else:
        imll=1
        gll2=diag(1./(2*arange(lmax)+1))

    # Check lmax
    lmax_union = min([imll.shape[0],min(min(b.shape[0] for b in ps.spectra.values()) for ps in [beam,pcls,noise,subpix] if isinstance(ps,PowerSpectra))])
    if lmax_union < lmax and is_mpi_master(): print "Warning: Lowering lmax from %i to %i because of insufficient data."%(lmax,lmax_union)
    lmax = lmax_union
    for ps in ['beam','pcls','noise','subpix']:
        exec("if isinstance(%s,PowerSpectra): %s = %s.sliced(0,lmax)"%((ps,)*3))
    imll = imll[:lmax,:lmax]
    if gll2 is not None: gll2 = gll2[:lmax,:lmax]
    ells = arange(lmax)
    todl = ells*(ells+1)/2/pi

    #Equation (4), the per detector signal estimate
    if (is_mpi_master()): print "Calculating per-detector signal..."
    hat_cls_det = PowerSpectra(ells=bin(ells))
    for (a,b) in pairs(maps): 
        hat_cls_det[(a,b)] = bin(todl*(dot(imll,pcls[(a,b)]) - (noise[a] if a==b else 0) - subpix[a,b])/beam[(a,b)]/H.pixwin(2048)[:lmax]**2)

    # Do per detector calibration
    if (str2bool(params.get('do_calibration',True))):
        if (is_mpi_master()): print "Fitting calibration factors..."
        calib = dict(flatten([[(m,a) for (m,[(_,a)]) in hat_cls_det.calibrated([m for m in maps if m.fr==fr], bin(slice(150,300))) if m.fr==fr] for fr in freqs]))
        for (a,b) in pairs(maps): hat_cls_det[(a,b)] *= calib[a]*calib[b]
    else: calib = defaultdict(lambda: 1)


    # Equation (6), the per frequency signal estimate
    if (is_mpi_master()): print "Calculating signal..."
    hat_cls_freq = PowerSpectra(ells=bin(ells),binning=params["binning"])
    for (alpha,beta) in pairs(freqs):
        hat_cls_freq[(alpha,beta)] = sum(
                hat_cls_det[(a,b)]*weight[(a,b)]
                for (a,b) in pairs(maps) if (a.fr, b.fr) in [(alpha,beta),(beta,alpha)]
            )/sum(
                weight[(a,b)] 
                for (a,b) in pairs(maps) if (a.fr, b.fr) in [(alpha,beta),(beta,alpha)]
            )
    


    # The fiducial model for the mask deconvolved Cl's which gets used in the covariance
    if str2bool(params.get("get_covariance",False)):
        if (is_mpi_master()): print "Calculating fiducial signal..."
        fid_cls = PowerSpectra(ells=ells)
        for (a,b) in pairs(maps):
            fid_cls[(a,b)] = smooth(dot(imll,pcls[(a,b)])/H.pixwin(2048)[:lmax]**2*(ells+2)**2,window_len=50)/(ells+2)**2*calib[a]*calib[b]
            fid_cls[(a,b)][:20] = 1000/arange(1,21)**2*2*pi

    
    if str2bool(params.get("get_covariance",True)):
        if (is_mpi_master()): print "Calculating detector covariance..."
        def entry(((a,b),(c,d))):
            if weight[(a,b)]*weight[(c,d)]!=0:
                print "Calculating the "+str(((str(a),str(b)),(str(c),str(d))))+" entry."
                # Equation (5)
                pclcov = (lambda x: x+x.T)(outer(fid_cls[(a,c)],fid_cls[(b,d)]) + outer(fid_cls[(a,d)],fid_cls[(b,c)]))*gll2/2
                # Equation (7)
                return (((a,b),(c,d)),dot(bin(todl*(imll/transpose([beam[(a,b)]])),axis=0),dot(pclcov,bin(todl*(imll.T/(beam[(c,d)])),axis=1)))*calib[a]*calib[b]*calib[c]*calib[d])
            else:
                return (((a,b),(c,d)),0)

        hat_cls_det.cov=SymmetricTensorDict(mpi_map(entry,pairs(pairs(maps))),rank=4)
        
        # Equation (9)
        if (is_mpi_master()): 
            print "Calculating covariance..."
            for ((alpha,beta),(gamma,delta)) in pairs(pairs(freqs)):
                if (is_mpi_master()): print "Calculating the "+str(((alpha,beta),(gamma,delta)))+" entry."

                abcds=[((a,b),(c,d)) 
                       for (a,b) in pairs(maps) for (c,d) in pairs(maps)
                       if (a.fr,b.fr) in [(alpha,beta),(beta,alpha)] and (c.fr,d.fr) in [(gamma,delta),(delta,gamma)]]
                
                hat_cls_freq.cov[((alpha,beta),(gamma,delta))] = \
                    sum(weight[(a,b)]*weight[(c,d)]*hat_cls_det.cov[((a,b),(c,d))] for ((a,b),(c,d)) in abcds) \
                    / sum(weight[(a,b)]*weight[(c,d)] for ((a,b),(c,d)) in abcds)
            
    if (is_mpi_master()): hat_cls_freq.save_as_matrix(params["signal"])

