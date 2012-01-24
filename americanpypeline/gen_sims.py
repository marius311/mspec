#!/usr/bin/env python

from americanpypeline import *
import sys, os
from numpy import *
import healpy as H

"""
Given a beams directory, a signal, (optional) noise, and Nside, this script generates map simulations 
"""

params = dict(arg.split("=") for arg in sys.argv[1:])

cmb = H.read_map(params["cmb"])
nside = H.npix2nside(alen(cmb))
lmax  = 3*nside
beams = load_beams({"beams":params["beams"],"lmax":lmax})

for (m,b) in beams.items():
    filename = os.path.join(params["maps"],str(m)+"_sim.fits")
    print "Simulating "+filename+"..."
    H.write_map(filename, H.alm2map(smooth_alms(H.map2alm(cmb),b),nside)/1e6)