from americanpypeline import *
import pypico
from utils import *
from numpy import *
import os

def cmb(p,ps):
    cmbfunc = pypico.pico if p.get('use_pico',False) else pypico.camb
    cmb = hstack([[0],cmbfunc(**p)[0][:p["lmax"]-1]])
    return PowerSpectra({k:cmb for k in ps})

def cmb_wmap(p,ps):
    if "cmb_wmap" in p.keys(): cmb=p["cmb_wmap"]
    else: p["cmb_wmap"]=cmb=loadtxt(os.path.join(AProotdir,"dat/external/bestfit_lensedCls.dat"))[:p["lmax"],1]
    return PowerSpectra({k:cmb for k in ps})
        
def dg_po(p,maps):
    ps = PowerSpectra()
    norm_fr = p["norm_freq"]["dusty"]
    norm_ell = p["norm_ell"]
    ells = arange(p["lmax"])
    todl = (ells*(ells+1))/(2*pi)
    for (fr1,fr2) in pairs(maps):
        efr1, efr2, alpha, Amp = p["eff_freqs"]["dusty"][fr1], p["eff_freqs"]["dusty"][fr2], p["alpha_dg_po"], p["A_dg_po"]
        #Decoherence between 143/217 and whatever higher frequency we do the cleaning with
        r=1
        if (fr1!=fr2):
            if '143' in [fr1,fr2]: r=p["r_dg_po_143"]
            elif '217' in [fr1,fr2]: r=p["r_dg_po_217"]
        ps[(fr1,fr2)] += r*(Amp/norm_ell)*(efr1*efr2/norm_fr**2)**alpha \
                         /dBdT(efr1,norm_fr)/dBdT(efr2,norm_fr) \
                         *todl
    return ps
    
    
def radio_po(p,maps):
    ps = PowerSpectra()
    norm_fr = p["norm_freq"]["radio"]
    norm_ell = p["norm_ell"]
    ells = arange(p["lmax"])
    todl = (ells*(ells+1))/(2*pi)
    for (fr1,fr2) in pairs(maps):
        efr1, efr2, alpha, Amp = p["eff_freqs"]["radio"][fr1], p["eff_freqs"]["radio"][fr2], p["alpha_radio_po"], p["A_radio_po"]
        ps[(fr1,fr2)] = (Amp/norm_ell)*(efr1*efr2/norm_fr**2)**alpha \
                         /dBdT(efr1,norm_fr)/dBdT(efr2,norm_fr) \
                         *todl
    return ps
    
    
def fg_from_amps(p,ps,prefix,ellshape):
    return PowerSpectra({(m1,m2): p["_".join([prefix,m1,m2])]*ellshape for (m1,m2) in ps})


