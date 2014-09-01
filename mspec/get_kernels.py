#!/usr/bin/env python

from mspec import *
import healpy as H
import sys, os, re, os.path as osp
from numpy import *
from utils import *
from numpy.linalg import inv
import cPickle
from multiprocessing.dummy import Pool
import psutil
from mpi4py import MPI
import h5py


def get_kernels(masks,
                kernels,
                lmax,
                masks_forgll=None,
                mask_name_transform=default_mask_name_transform,
                clobber=False,
                **kwargs):

    def get_alms((m1,m2)):
        mspec_log('Getting alms for '+str((m1,m2)))
        return ((m1,m2),
                H.map2alm(H.read_map(m1,verbose=False)*(1 if m2 is None else H.read_map(m2,verbose=False)),
                          lmax=min(3*2048-1,2*lmax)))

    def get_cls((m12,m23)):
        mspec_log('Getting cls for '+str((m12,m23)))
        return ((m12,m23), H.alm2cl(alms[m12],alms[m23]))


    def get_kernel((k,ks,ktype)):

        mspec_log('Doing %s'%str(ks))
        mlls = qk.get_mlls(cls[k])
        def getk((a,b,c,d)): return hdf[ktype][str(ks)][str(((a,b),(c,d)))]

        if ktype=='imll':
            #invert 2x2 EE-BB block
            imllPP = inv(hstack(hstack([[mlls['EE','EE'][2:lmax,2:lmax], mlls['EE','BB'][2:lmax,2:lmax]],
                                        [mlls['EE','BB'][2:lmax,2:lmax], mlls['EE','EE'][2:lmax,2:lmax]]])))

            getk('TTTT')[: ,: ] = inv(mlls['TT','TT'])
            getk('TETE')[2:,2:] = inv(mlls['TE','TE'][2:,2:])
            getk('EBEB')[2:,2:] = inv(mlls['EB','EB'][2:,2:])
            getk('EEEE')[2:,2:] = imllPP[:lmax-2,:lmax-2]
            getk('EEBB')[2:,2:] = imllPP[:lmax-2,lmax-2:]

        elif ktype=='gll':
            for (a,b),v in mlls.items():
                getk(a+b)[:] = v/(2*arange(v.shape[0])+1)
        
        mspec_log(str(psutil.virtual_memory()))


    masks = {k:mask_name_transform(k) for k in masks.values()}
    if masks_forgll:
        masks_forgll = {k:mask_name_transform(k) for k in masks_forgll.values()}
    if is_mpi_master() and not osp.exists(osp.dirname(kernels)): os.makedirs(kernels)
    pool=Pool(get_num_threads()/2)

    #alms
    specs, specs_forgll = [(m,None) for m in masks], pairs(masks_forgll)
    mspec_log('Getting kernels for:\n'+'\n'.join(['  '+str(s) for s in (specs+specs_forgll)]),rootlog=True)
    alms = dict(mpi_map2(get_alms,specs+specs_forgll))

    #cls
    if is_mpi_master():
        cls = dict(pool.map(get_cls,pairs(specs)+pairs(specs_forgll)))
    if not is_mpi_master() or get_mpi_size()==1:
        mspec_log('Precompting Wigner 3j quantities...')
        qk = quickkern(lmax,pool=pool)
    cls = mpi_consistent(cls if is_mpi_master() else None)

    #kernels
    work  = [(((m1,None),(m2,None)),(n1,n2),'imll') 
             for ((m1,n1),(m2,n2)) in pairs(masks.items())]
    work += [(((m1,m2),(m3,m4)),((n1,n2),(n3,n4)),'gll')
             for (((m1,n1),(m2,n2)),((m3,n3),(m4,n4))) in pairs(pairs(masks_forgll.items()))]
    
    hdf = h5py.File(kernels,mode='w',driver='mpio', comm=MPI.COMM_WORLD)
    try:
        #set up groups, which all processes must do                                
        for t in ['imll','gll']: hdf.create_group(t)
        for k,kn,ktype in work:
            g = hdf[ktype].create_group(str(kn))
            if ktype=='imll':
                for ks in SymmetricTensorDict({kn:None},rank=2): 
                    if ks!=kn: hdf[ktype][str(ks)] = g
                syms = {'TTTT':[], 'TETE':['ETET','TBTB','BTBT'], 'EBEB':['BEBE'], 'EEEE':['BBBB'], 'EEBB':['BBEE']}
                for (a,b,c,d),sym in syms.items(): 
                    ds = g.create_dataset(str(((a,b),(c,d))),(lmax,lmax))
                    for (x,y,z,w) in sym: g[str(((x,y),(z,w)))] = ds
            elif ktype=='gll':
                for ks in SymmetricTensorDict({kn:None},rank=4): 
                    if ks!=kn: hdf[ktype][str(ks)] = g
                for (a,b,c,d) in ['TTTT','TETE','EEEE','EBEB','EEBB']:
                    g.create_dataset(str(((a,b),(c,d))),(lmax,lmax))

        #compute kernels
        mpi_map2(get_kernel,work,pool=pool)

    finally:
        hdf.close()



class quickkern(object):

    import quickbeam.maths as maths
    cf_from_cl = maths._mathsc.wignerd_cf_from_cl
    cl_from_cf = maths._mathsc.wignerd_cl_from_cf
    init_gauss_legendre_quadrature = maths._mathsc.init_gauss_legendre_quadrature

    def __init__(self, lmax, ngl=None, ss=[(0,0),(0,2),(2,2)], pool=None):
        self.lmax = lmax
        self.ngl = ngl = 3*lmax/2+1 if ngl is None else ngl
        self.ss = ss
        self.zvec, self.wvec = self.init_gauss_legendre_quadrature(ngl)
        _map = map if pool is None else pool.map
        self.plzi = dict(zip(ss,_map(lambda (s1,s2): self.cf_from_cl(s1, s2, lmax, ngl, lmax-1, self.zvec, diag(2*arange(lmax)+1).flatten()).reshape(lmax,ngl),ss)))

    def get_mll(self,wl,s1=0,s2=0):
        assert (s1,s2) in self.ss
        lmax_wl = wl.size
        fzi = self.cf_from_cl(0, 0, 1,self.ngl,lmax_wl-1,self.zvec,(2*arange(lmax_wl)+1.)*wl/8/pi)
        plzi = self.plzi[s1,s2]*fzi
        mll = self.cl_from_cf(s1, s2, self.lmax, self.ngl, self.lmax-1, self.zvec, self.wvec, plzi.flatten()).reshape(self.lmax,self.lmax).T
        return mll

    def get_mlls(self,wl):
        lmax_wl = wl.size

        m00   = self.get_mll(wl,0,0)
        m22_1 = self.get_mll(wl,2,2)
        m22_2 = self.get_mll(wl*(-1)**(arange(lmax_wl)),2,2)
        m02_1 = self.get_mll(wl,0,2)
        m02_2 = self.get_mll(wl*(-1)**(arange(lmax_wl)),0,2)

        minus1toL = (-1)** (arange(self.lmax).reshape((1,self.lmax)) + arange(self.lmax).reshape((self.lmax,1)))

        return {('TT','TT'):m00,
                ('TE','TE'):(m02_1 + m02_2 * minus1toL)/2,
                ('EE','EE'):(m22_1 + m22_2 * minus1toL)/2,
                ('EE','BB'):(m22_1 - m22_2 * minus1toL)/2,
                ('EB','EB'):m22_2 * minus1toL}
