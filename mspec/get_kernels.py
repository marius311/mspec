#!/usr/bin/env python

from mspec import *
import healpy as H
import sys, os, re, os.path as osp
from numpy import *
from utils import *
from numpy.linalg import inv


def get_kernels(masks,
                kernels,
                lmax,
                mask_name_transform=default_mask_name_transform,
                **kwargs):

    def dowork((((m1,m2),(m3,m4)),mtype,output)):
        print 'Process %i is doing %s'%(get_mpi_rank(),output)

        m12 = H.read_map(m1) * (1 if m2 is None else H.read_map(m2))
        m34 = H.read_map(m3) * (1 if m4 is None else H.read_map(m4))
        
        mll = get_mll(lmax,H.anafast(m12,m34,lmax=min(3*H.npix2nside(m12.size)-1,2*lmax)))

        if mtype=='imll': 
            np.save(output,inv(mll))
        elif mtype=='gll':
            np.save(output,mll.T/(2*arange(mll.shape[0])+1))
        else:
            raise ValueError(mtype)


    masks = {m:mask_name_transform(m) for m in set(masks.values())}

    work =  [(((m1,None),(m2,None)), 'imll',  osp.join(kernels,'imll__%s__X__%s'%(n1,n2)))              for (m1,n1),(m2,n2) in pairs(masks.items())]
    work += [(((m1,m2),(m3,m4)),     'gll' , osp.join(kernels,'gll__%s__%s__X__%s__%s'%(n1,n2,n3,n4))) for ((m1,n1),(m2,n2)),((m3,n3),(m4,n4)) in pairs(pairs(masks.items()))]

    if is_mpi_master(): print 'Getting %i kernels...'%len(work)
    mpi_map(dowork,work)



import quickbeam.maths as maths
cf_from_cl = maths._mathsc.wignerd_cf_from_cl
cl_from_cf = maths._mathsc.wignerd_cl_from_cf
init_gauss_legendre_quadrature = maths._mathsc.init_gauss_legendre_quadrature

def get_mll(lmax,wl,s1=0,s2=0):
    lmax_wl = wl.size
    ngl = 3*lmax/2+1
    zvec, wvec = init_gauss_legendre_quadrature(ngl)
    
    fzi = cf_from_cl(s1, s2, 1,ngl,lmax_wl-1,zvec,(2*arange(lmax_wl)+1)*wl/8/pi)
    
    #about 1/3 of time is spent on this line which could be precomputed:
    plzi = cf_from_cl(s1, s2, lmax, ngl, lmax-1, zvec, diag(2*arange(lmax)+1).flatten()).reshape(lmax,ngl)
    
    plzi *= fzi
    mll = cl_from_cf(s1, s2, lmax, ngl, lmax-1, zvec, wvec, plzi.flatten()).reshape(lmax,lmax).T
    
    return mll

