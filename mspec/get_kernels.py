#!/usr/bin/env python

from mspec import *
import healpy as H
import sys, os, re, os.path as osp
from numpy import *
from utils import *
from numpy.linalg import inv
import cPickle

def get_kernels(masks,
                kernels,
                lmax,
                mask_name_transform=default_mask_name_transform,
                **kwargs):

    def dowork((((m1,m2),(m3,m4)),mtype,output)):
        print 'Process %i is doing %s'%(get_mpi_rank(),output)

        m12 = H.read_map(m1) * (1 if m2 is None else H.read_map(m2))
        m34 = H.read_map(m3) * (1 if m4 is None else H.read_map(m4))

        mlls = get_mlls(lmax,H.anafast(m12,m34,lmax=min(3*H.npix2nside(m12.size)-1,2*lmax)))

        with open(output,"w") as f:
            if mtype=='imll':
                imllPP = zeros((2*lmax,)*2)
                s=hstack([arange(2,lmax),arange(lmax+2,2*lmax)])
                imllPP[ix_(s,s)] = inv(hstack(hstack([[mlls['EE','EE'][2:lmax,2:lmax], mlls['EE','BB'][2:lmax,2:lmax].T],
                                                      [mlls['EE','BB'][2:lmax,2:lmax], mlls['BB','BB'][2:lmax,2:lmax]  ]])))
                imllTE = zeros((lmax,)*2)
                imllTE[2:,2:]=inv(mlls['TE','TE'][2:,2:])

                imllEB = zeros((lmax,)*2)
                imllEB[2:,2:]=inv(mlls['EB','EB'][2:,2:])

                imll = {(('T','T'),('T','T')):inv(mlls['TT','TT']),
                        (('T','E'),('T','E')):imllTE,
                        (('E','B'),('E','B')):imllEB}
                imll.update({[[(('E','E'),('E','E')),(('E','E'),('B','B'))],
                              [(('B','B'),('E','E')),(('B','B'),('B','B'))]][i][j]: [split(x,2,axis=1) for x in split(imllPP,2)][i][j]
                             for i in range(2) for j in range(2)})


                cPickle.dump(imll,f,protocol=2)

            elif mtype=='gll':
                cPickle.dump({k:v/(2*arange(v.shape[0])+1) for k,v in mlls.items()},f,protocol=2)
            else:
                raise ValueError(mtype)

    if type(masks)!=dict: masks={'mask':masks}
    masks = {m:mask_name_transform(m) for m in set(masks.values())}

    work =  [(((m1,None),(m2,None)), 'imll', osp.join(kernels,'imll__%s__X__%s'%(n1,n2)))              for (m1,n1),(m2,n2) in pairs(masks.items())]
    work += [(((m1,m2),(m3,m4)),     'gll' , osp.join(kernels,'gll__%s__%s__X__%s__%s'%(n1,n2,n3,n4))) for ((m1,n1),(m2,n2)),((m3,n3),(m4,n4)) in pairs(pairs(masks.items()))]

    if is_mpi_master(): print 'Getting %i kernels...'%len(work)
    mpi_map(dowork,work)



import quickbeam.maths as maths
cf_from_cl = maths._mathsc.wignerd_cf_from_cl
cl_from_cf = maths._mathsc.wignerd_cl_from_cf
init_gauss_legendre_quadrature = maths._mathsc.init_gauss_legendre_quadrature

def get_mll(lmax,wl,s1=0,s2=0,ngl=None):
    lmax_wl = wl.size
    if ngl is None: ngl = 3*lmax/2+1
    zvec, wvec = init_gauss_legendre_quadrature(ngl)

    fzi = cf_from_cl(0, 0, 1,ngl,lmax_wl-1,zvec,(2*arange(lmax_wl)+1.)*wl/8/pi)

    #about 1/3 of time is spent on this line which could be precomputed:
    plzi = cf_from_cl(s1, s2, lmax, ngl, lmax-1, zvec, diag(2*arange(lmax)+1).flatten()).reshape(lmax,ngl)

    plzi *= fzi
    mll = cl_from_cf(s1, s2, lmax, ngl, lmax-1, zvec, wvec, plzi.flatten()).reshape(lmax,lmax).T

    return mll


def get_mlls(lmax,wl):
    lmax_wl = wl.size

    m00 = get_mll(lmax,wl,0,0)

    m22_1 = get_mll(lmax,wl,2,2)
    m22_2 = get_mll(lmax,wl*(-1)**(arange(lmax_wl)),2,2)

    m02_1 = get_mll(lmax,wl,0,2)
    m02_2 = get_mll(lmax,wl*(-1)**(arange(lmax_wl)),0,2)

    minus1toL = (-1)** (arange(lmax).reshape((1,lmax)) + arange(lmax).reshape((lmax,1)))

    return {('TT','TT'):m00,
            ('TE','TE'):(m02_1 + m02_2 * minus1toL)/2,
            ('EE','EE'):(m22_1 + m22_2 * minus1toL)/2,
            ('BB','BB'):(m22_1 + m22_2 * minus1toL)/2,
            ('EE','BB'):(m22_1 - m22_2 * minus1toL)/2,
            ('EB','EB'):m22_2 * minus1toL}
