#!/usr/bin/env python

import mcmc
from americanpypeline import *
import sys
import pycamb
from scipy.linalg import cho_factor, cho_solve
import skymodel

    
def camb_derived(p):
    h = pycamb.theta2hubble(p["theta"],p["omegabh2"],p["omegach2"],p["omegak"],p["omeganuh2"],p["w"],p["massless_neutrinos"],p["massive_neutrinos"])/100.
    p["H0"]=100*h
    p["omegab"]=p["omegabh2"]/h**2
    p["omegac"]=p["omegach2"]/h**2
    p["omeganu"]=p["omeganuh2"]/h**2
    p["omegav"]=1-p["omegab"]-p["omegac"]-p["omeganu"]
    p["As"]=exp(p["logA"])*10**(-10)
    
def lnl(p):
    model = get_skymodel(p).sliced(p["lmin"],p["lmax"]).binned(p["mcmc_binning"]).get_as_matrix(ell_blocks=True).spec[:,1]  
    dcl = model - p["signal_mat"].spec 
    return dot(dcl,cho_solve(p["signal_mat"].cov,dcl))/2
    
def get_skymodel(p):
    ells=arange(p["lmax"])
    dgpo = lambda p, ps: skymodel.fg_from_amps(p,ps,"dgpo",ells*(ells+1)/(p["norm_ell"]*(p["norm_ell"]+1)))
    radio = lambda p, ps: skymodel.fg_from_amps(p,ps,"radio",ells*(ells+1)/(p["norm_ell"]*(p["norm_ell"]+1)))
    comps = [skymodel.cmb,dgpo,radio]
    return PowerSpectra.sum([comp(p,pairs(p["signal"].get_maps())) for comp in comps])
        
def init(p):
    p["warn"]=False
    if ("pico_datafile" in p):
        print "Initializing PICO..."
        pycamb.picoinit(p["pico_datafile"],verbose=p.get("pico_verbose",True))
    if ("wmadata_dir" in p):
        print "Initializing WMAP..."
        pywmap.wmapinit(p["wmap_data_dir"])
    
    print "Loading signal and covariance..."
    p["ells"]=arange(p["lmin"],p["lmax"])
    p["signal"]=load_clean_calib_signal(p).sliced(p["lmin"],p["lmax"]).binned(p["mcmc_binning"])
    p["signal_mat"]=p["signal"].get_as_matrix(ell_blocks=True)
    p["signal_mat"]=namedtuple("SpecCov",["spec","cov"])(p["signal_mat"].spec[:,1],cho_factor(p["signal_mat"].cov))
    

if __name__=="__main__":
    
    if (len(sys.argv) != 2): 
        print "Usage: python signal_to_params.py parameter_file.ini"
        sys.exit()

    params = read_AP_ini(sys.argv[1])
    
    mcmc.mpi_mcmc(params, lnl=lnl, init_fn=init, derived_fn=camb_derived)

    
    