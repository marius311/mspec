#!/usr/bin/env python

from mcmc import *
from americanpypeline import *
import sys
import pypico
from scipy.linalg import cho_factor, cho_solve
import skymodel


@depends("theta","omegabh2","omegach2","omegak","omeganuh2","w","massless_neutrinos","massive_neutrinos")
def camb_derived(p):
    h = pypico.theta2hubble(p["theta"],p["omegabh2"],p["omegach2"],p["omegak"],p["omeganuh2"],p["w"],p["massless_neutrinos"],p["massive_neutrinos"])/100.
    return {"H0":100*h,
            "omegab":p["omegabh2"]/h**2,
            "omegac":p["omegach2"]/h**2,
            "omeganu":p["omeganuh2"]/h**2,
            "omegav":1-(p["omegabh2"]+p["omegach2"]+p["omeganuh2"])/h**2,
            "As":exp(p["logA"])*10**(-10)}


def lnl(p):
    model = get_skymodel(p) 
#    model.spectra[('143','143')] *= (1 + sum([p['calib143(%i)'%i] for i in range(p["beam_pca"].shape[1])] * p["beam_pca"],axis=1))
    model = model.sliced(p["binning"](slice(p["lmin"],p["lmax"]))).get_as_matrix(ell_blocks=True).spec[:,1]  
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
        pypico.picoinit(p["pico_datafile"],verbose=p.get("pico_verbose",True))
    if ("wmadata_dir" in p):
        print "Initializing WMAP..."
        pywmap.wmapinit(p["wmap_data_dir"])
    
    print "Loading signal and covariance..."
    p["ells"]=arange(p["lmin"],p["lmax"])
    p["signal"]=load_signal(p).to_dl().sliced(p['binning'](slice(p["lmin"],p["lmax"])))
    p["signal_mat"]=p["signal"].get_as_matrix(ell_blocks=True)
    p["signal_mat"]=namedtuple("SpecCov",["spec","cov"])(p["signal_mat"].spec[:,1],cho_factor(p["signal_mat"].cov))
    
    p["beam_pca"] = loadtxt(p["beam_pca"])[:p["lmax"],2:3]
    
if __name__=="__main__":
    
    if (len(sys.argv) != 2): 
        print "Usage: python signal_to_params.py parameter_file.ini"
        sys.exit()

    bestfit(read_AP_ini(sys.argv[1]), lnl=lnl, init_fn=init, derived_fn=camb_derived)

    
    
