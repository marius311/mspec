#!/usr/bin/env python

"""
 This script computes an estimate of the signal power spectrum 
 and the covariance of the estimate given some pseudo-powerspectra 
 computed by maps_to_pcl.py
 
 Paramter file:
 
     lmax - Maximum ell when calculating the powerspectrum.
     pcls - The folder where the pseudo-powerspectra are.
     signal - The binary files [signal]_estimate.dat and [signal]_covariance.dat will be created.
     
     Optional-
     
     mask - A mask to apply before calculating the powerspectrum, default None
     map_freq_regex/map_det_regex - Regular Expression that when applied to the map filename, the 
                                    first group returns the frequency/detector id. Defaults to 
                                    accepting names like '100-1a_W_TauDeconv_v47.fits'

""" 

from americanpypeline import *
import sys, os, re
from numpy import *
from numpy.linalg import norm
from collections import namedtuple, defaultdict
from utils import *
from l3l4sum import l3l4sum
l3l4sum = l3l4sum.l3l4_sum_single
from bisect import bisect_right


def main(params):
    
    params = read_AP_ini(params)
    lmax = params["lmax"]
    bin = params["pcl_binning"]
    ells = bin(arange(lmax))
    
    # Load mode coupling matrices
    if (params.get("mask")):
        if (is_mpi_master()): print "Loading mode coupling matrices..."
        imll = load_multi(params["mask"]+".imll")
        assert alen(imll)>=lmax, "The mode-coupling matrix has not been calculated to high enough lmax. Please run mask_to_mll.py again."
        imll = imll[:lmax,:lmax]
        if str2bool(params.get("get_covariance","F")): gll2 = load_multi(params["mask"]+".mll2")[:lmax,:lmax]/(2*arange(lmax)+1)
    else:
        imll=1
        gll2=diag(1./(2*arange(lmax)+1))
        
    # Could certainly do something smarter than this:
    dbmode = int(params["dlmode"])/int(max(ells[1:]-ells[:-1]))
    
    # The raw pseudo-C_ells
    if (is_mpi_master()): print "Loading pseudo-cl's..."
    pcls = load_pcls(params)
    freqs = params["freqs"].split()
    assert not set(freqs)-set(m.fr for m in pcls.get_maps()), "Couldn't find pseudo-cl's for all the desired frequencies: "+freqs
    maps = [m for m in pcls.get_maps() if m.fr in freqs]
    
    beam = load_beams(params) if params.get("beams",None) else {m:1 for m in maps}
    noise = load_noise(params) if params.get("noise",None) else {m:0 for m in maps}
    if str2bool(params.get("use_auto_spectra",'F')): 
        def weight(a,b): return 1
    else:
        def weight(a,b): return 0 if a==b else 1

    if set(maps) - set(beam.keys()):
        print "Warning: Missing beams for the following detectors: \n"+str(set(maps) - set(beam.keys()))+"\
               \nContinuing without those maps."
        maps = [m for m in maps if m in beam.keys()]
    
        
    #Equation (4), the per detector signal estimate
    print "Calculating per-detector signal..."
    hat_cls_det = PowerSpectra(ells=ells)
    for (a,b) in pairs(maps): hat_cls_det[(a,b)] = bin((dot(imll,pcls[(a,b)]) - (noise[a] if a==b else 0))/(beam[a]*beam[b]))

    # Do per detector calibration
    print "Fitting calibration factors..."
    calib = dict(flatten([[(m,a) for (m,[(_,a)]) in hat_cls_det.calibrated([m for m in maps if m.fr==fr], bin(slice(150,300))) if m.fr==fr] for fr in freqs]))
    for (a,b) in pairs(maps): hat_cls_det[(a,b)] *= calib[a]*calib[b]

    
    # Equation (6), the per frequency signal estimate
    if (is_mpi_master()): print "Calculating signal..."
    hat_cls_freq = PowerSpectra(ells=ells)
    for (alpha,beta) in pairs(freqs):
        hat_cls_freq[(alpha,beta)] = sum(
                hat_cls_det[(a,b)]*weight(a,b) 
                for (a,b) in pairs(maps) if a.fr==alpha and b.fr==beta
            )/sum(
                weight(a,b) 
                for (a,b) in pairs(maps) if a.fr==alpha and b.fr==beta
            )
    
    # The fiducial model for the mask deconvolved Cl's which gets used in the covariance
    if (str2bool(params.get("get_det_covariance",False)) or str2bool(params.get("get_covariance",False))):
        print "Calculating fiducial signal..."
        fid_cls = PowerSpectra(ells=ells)
        for (a,b) in pairs(maps): fid_cls[(a,b)] = bin(smooth(dot(imll,pcls[(a,b)]),window_len=50))*calib[a]*calib[b]

    if str2bool(params.get("get_det_covariance",False)):
        if (is_mpi_master()): print "Calculating detector covariance..."
        # Equation (7)
        for ((a,b),(c,d)) in pairs(pairs(maps)):
            if (is_mpi_master()): print "Calculating the "+str(((str(a),str(b)),(str(c),str(d))))+" entry."
            hat_cls_det.cov[((a,b),(c,d))] = \
                    l3l4sum(imll, fid_cls[(a,c)], fid_cls[(b,d)], fid_cls[(a,d)], fid_cls[(b,c)], gll2, dbmode) \
                    *outer(1/(beam[a]*beam[b]),1/(beam[c]*beam[d])) \
                    *calib[a]*calib[b]*calib[c]*calib[d]

    
    if str2bool(params.get("get_covariance",True)):
        if (is_mpi_master()): print "Calculating covariance..."
        
        # Equation (7)
        for ((alpha,beta),(gamma,delta)) in pairs(pairs(freqs)):
            if (is_mpi_master()): print "Calculating the "+str(((alpha,beta),(gamma,delta)))+" entry."
            def term(abcds):
                s = sum(
                    weight(a,b)*weight(c,d)
                    *l3l4sum(imll, fid_cls[(a,c)], fid_cls[(b,d)], fid_cls[(a,d)], fid_cls[(b,c)], gll2, dbmode)
                    *sum(outer(1/(beam[a]*beam[b]),1/(beam[c]*beam[d])) for ((a,b),(c,d)) in syms)
                    *calib[a]*calib[b]*calib[c]*calib[d]
                    for (((a,b),(c,d)),syms) in abcds
                )
                return s
                
            abcds=[(((a,b),(c,d)),[((a,b),(c,d)),((b,a),(c,d)),((a,b),(d,c)),((b,a),(d,c))]) 
                   for (a,b) in pairs(maps) for (c,d) in pairs(maps)
                   if a.fr==alpha and b.fr==beta and c.fr==gamma and d.fr==delta]
            
            s = sum(mpi_map(term,partition(abcds,get_mpi_size())),axis=0)
            s /= sum(weight(a,b)*weight(c,d)*len(syms) for (((a,b),(c,d)),syms) in abcds)
            
            hat_cls_freq.cov[((alpha,beta),(gamma,delta))]=s
        
    
    hat_cls_freq.save_as_matrix(params["signal"])


if __name__=="__main__": 
    
    if (len(sys.argv) == 2): main(sys.argv[1])
    else: print "Usage: python mask_to_mll.py parameter_file.ini"
