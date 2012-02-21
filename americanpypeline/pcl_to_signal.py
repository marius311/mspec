#!/usr/bin/env python

"""
 This script computes an estimate of the signal power spectrum 
 and the covariance of the estimate given some pseudo-powerspectra 
 computed by maps_to_pcl.py
""" 

from americanpypeline import *
import sys, os, re
from numpy import *
from numpy.linalg import norm
from collections import namedtuple, defaultdict
from utils import *
import utils
from bisect import bisect_right



if __name__=="__main__": 
    
    if (len(sys.argv) != 2): 
        print "Usage: python mask_to_mll.py parameter_file.ini"
        sys.exit()

    utils.NOMPI = True    
    
    params = read_AP_ini(sys.argv[1])
    lmax = params["lmax"]
    bin = params["binning"]
    ells = arange(lmax)
    
    # Load stuff
    if (is_mpi_master()): print "Loading pseudo-cl's, beams, and noise..."
    pcls = load_pcls(params)
    beam = load_beams(params) if params.get("beams",None) else defaultdict(lambda: 1)
    noise = load_noise(params) if params.get("noise",None) else defaultdict(lambda: 0)

    freqs = params.get("freqs")
    maps = set(m for m in pcls.get_maps() if (m.fr in freqs if freqs!=None else True))
    if params.get("beams",None): maps&=set(beam.get_maps())
    if params.get("noise",None): maps&=set(noise.get_keys())
    freqs = set(m.fr for m in maps)
    
    if (is_mpi_master()): print "Found pseudo-cl's, beams, and noise for: "+str(sorted(maps))

    if str2bool(params.get("use_auto_spectra",'F')): weight = lambda a,b: 1
    else: weight = lambda a,b: 0 if a==b else 1

    # Load mode coupling matrices
    if (params.get("mask")):
        if (is_mpi_master()): print "Loading mode coupling matrices..."
        imll = load_multi(params["mask"]+".imll")
        assert alen(imll)>=lmax, "The mode-coupling matrix has not been calculated to high enough lmax. Please run mask_to_mll.py again."
        imll = imll[:lmax,:lmax]
        if str2bool(params.get("get_covariance",False)): gll2 = load_multi(params["mask"]+".mll2")[:lmax,:lmax]/(2*arange(lmax)+1)
    else:
        imll=1
        gll2=diag(1./(2*arange(lmax)+1))

    #Equation (4), the per detector signal estimate
    print "Calculating per-detector signal..."
    hat_cls_det = PowerSpectra(ells=bin(ells))
    for (a,b) in pairs(maps): hat_cls_det[(a,b)] = bin((dot(imll,pcls[(a,b)]) - (noise[a] if a==b else 0))/(beam[(a,b)]))

    # Do per detector calibration
    if (str2bool(params.get('do_calibration',True))):
        print "Fitting calibration factors..."
        calib = dict(flatten([[(m,a) for (m,[(_,a)]) in hat_cls_det.calibrated([m for m in maps if m.fr==fr], bin(slice(150,300))) if m.fr==fr] for fr in freqs]))
        for (a,b) in pairs(maps): hat_cls_det[(a,b)] *= calib[a]*calib[b]
    else: calib = defaultdict(lambda: 1)

    
    # Equation (6), the per frequency signal estimate
    if (is_mpi_master()): print "Calculating signal..."
    hat_cls_freq = PowerSpectra(ells=bin(ells))
    for (alpha,beta) in pairs(freqs):
        hat_cls_freq[(alpha,beta)] = sum(
                hat_cls_det[(a,b)]*weight(a,b) 
                for (a,b) in pairs(maps) if a.fr==alpha and b.fr==beta
            )/sum(
                weight(a,b) 
                for (a,b) in pairs(maps) if a.fr==alpha and b.fr==beta
            )
    
    # The fiducial model for the mask deconvolved Cl's which gets used in the covariance
    if str2bool(params.get("get_covariance",False)):
        print "Calculating fiducial signal..."
        fid_cls = PowerSpectra(ells=ells)
        for (a,b) in pairs(maps): 
            fid_cls[(a,b)] = smooth(dot(imll,pcls[(a,b)])*(ells+2)**2,window_len=50)/(ells+2)**2*calib[a]*calib[b]
            fid_cls[(a,b)][:10] = 0

    
    if str2bool(params.get("get_covariance",True)):
        if (is_mpi_master()): print "Calculating detector covariance..."
        for ((a,b),(c,d)) in pairs(pairs(maps)):
            if weight(a,b)*weight(c,d)!=0:
                if (is_mpi_master()): print "Calculating the "+str(((str(a),str(b)),(str(c),str(d))))+" entry."
                # Equation (5)
                pclcov = (lambda x: x+x.T)(outer(fid_cls[(a,c)],fid_cls[(b,d)]) + outer(fid_cls[(a,d)],fid_cls[(b,c)]))*gll2/2
                # Equation (7)
                hat_cls_det.cov[((a,b),(c,d))] = \
                    dot(bin(imll/transpose([beam[(a,b)]]),axis=0),dot(pclcov,bin(imll.T/(beam[(c,d)]),axis=1))) \
                    *calib[a]*calib[b]*calib[c]*calib[d]
            else:
                hat_cls_det.cov[((a,b),(c,d))] = 0
        
        # Equation (9)
        if (is_mpi_master()): print "Calculating covariance..."
        for ((alpha,beta),(gamma,delta)) in pairs(pairs(freqs)):
            if (is_mpi_master()): print "Calculating the "+str(((alpha,beta),(gamma,delta)))+" entry."

            abcds=[((a,b),(c,d)) 
                   for (a,b) in pairs(maps) for (c,d) in pairs(maps)
                   if a.fr==alpha and b.fr==beta and c.fr==gamma and d.fr==delta]
            
            hat_cls_freq.cov[((alpha,beta),(gamma,delta))] = \
                sum(weight(a,b)*weight(c,d)*hat_cls_det.cov[((a,b),(c,d))] for ((a,b),(c,d)) in abcds) \
                / sum(weight(a,b)*weight(c,d) for ((a,b),(c,d)) in abcds)

            
    
    hat_cls_freq.save_as_matrix(params["signal"])

