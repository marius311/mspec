from collections import namedtuple, defaultdict
from americanpypeline import *
from americanpypeline.utils import *
import sys, os, re, gc
from numpy import *
import healpy as H

cmbdir = '/project/projectdirs/planck/data/ctp3/mc_cmb'
noisedir = '/project/projectdirs/planck/data/ctp3/mc_noise'
outdir = '/global/homes/m/marius/workspace/americanpypeline/test/1000ctp/pcls'


dat = defaultdict(lambda: namedtuple('dat',['cmb','noise1','noise2']))
getidx = lambda f: int(re.search('([0-9]{4,})',f).group())

for f in os.listdir(cmbdir): 
    dat[getidx(f)].cmb = os.path.join(cmbdir,f)
    
for f in os.listdir(noisedir): 
    if '1357' in f: dat[getidx(f)].noise1 = os.path.join(noisedir,f)
    elif '2468' in f: dat[getidx(f)].noise2 = os.path.join(noisedir,f)

mask = H.read_map('/global/homes/m/marius/workspace/americanpypeline/test/ctp3sim/cb6_gal_nest_ip_all_2048.fits')
lmax = 3000

def work((i,d)):
    outi = os.path.join(outdir,'%.5i'%i)
    if not os.path.exists(outi):
        print "Process "+str(get_mpi_rank())+" is doing realization "+str(i)
        cmb=H.read_map(d.cmb)
        noise = [H.read_map(n) for n in [d.noise1,d.noise2]]
        for n in noise: n[n==H.UNSEEN]=0
        alm1, alm2 = [H.map2alm(mask*(cmb+n)*1.654*1e6,lmax=lmax) for n in noise]
        cls = [alm2cl(*p) for p in pairs([alm1,alm2])]
        os.mkdir(outi)
        for ((i,j),cl) in zip([('0','0'),('0','1'),('1','1')],cls):
            savetxt(os.path.join(outi,'143-T-'+i+'__143-T-'+j),cl)
        gc.collect()

    
    
mpi_map(work,dat.items())


