from collections import namedtuple, defaultdict
from americanpypeline import *
import sys, os, re
from numpy import *
import healpy as H

cmbdir = '/project/projectdirs/planck/data/ctp3/mc_cmb'
noisedir = '/project/projectdirs/planck/data/ctp3/mc_noise'

params = read_AP_ini(sys.argv[1])
mask = H.read_map(params["mask"])
lmax = int(params["lmax"])

dat = defaultdict(lambda: namedtuple(['cmb','noise1','noise1']))
getidx = lambda f: re.search('([0-9]+)',f).group()

for f in os.listdir(cmbdir): 
    dat[getidx(f)].cmb = os.path.join(cmbdir,f)
    
for f in os.listdir(noisedir): 
    if '1357' in f: dat[getidx(f)].noise1 = os.path.join(cmbdir,f)
    elif '2468' in f: dat[getidx(f)].noise2 = os.path.join(cmbdir,f)

def work(i,d):
    if not os.path.exists(os.path.join(outdir,i)):
        print "Process "+str(get_mpi_rank())+" is doing "+i
        cmb=H.read_map(d.cmb)
        noise = [H.read_map(n) for n in [d.noise1,d.noise2]]
        alm1, alm2 = [H.map2alm(mask*(cmb+n)*1.654*1e6,lmax=lmax) for n in noise]
        cls = [AP.alm2cl(*p) for p in pairs([alm1,alm2])]
        for ((i,j),cl) in zip([('a','a'),('a','b'),('b','b')],cls):
            savetxt(os.path.join(outdir,i,'143-T-1'+i+'___-143-T-1'+j),cl)

    
    
mpi_map(work,dat)


