from ast import literal_eval
from bisect import bisect_right
from collections import namedtuple
from matplotlib.pyplot import *
from numpy import *
from numpy.linalg import norm
from scipy.optimize import fmin
from utils import *
import healpy as H
import sys, os, re, gc


freqs = ['143','217','353']

MapID = namedtuple("MapID", ["fr","type","id"])
MapID.__str__ = lambda self: "-".join(self)  


def cov_syms(((a,b),(c,d))):
    return set([((a,b),(c,d)),((b,a),(c,d)),((a,b),(d,c)),((b,a),(d,c)),
            ((c,d),(a,b)),((c,d),(b,a)),((d,c),(a,b)),((d,c),(b,a))])
    
class SymmetricTensorDict(dict):
    """
    A dictionary where keys are either (a,b) or ((a,b),(c,d)). 
    When a key is added, other keys which have the same value because of symmetry are also added. 
    The symmetry properties assumed are:
        (a,b) - Fully symmetric
        ((a,b),(c,d)) - The symmetry properties of the C_ell covariance (not quite fully symmetric)
    """
    
    def __init__(self, *args, **kwargs):
        self.rank=kwargs.pop("rank",2)
        assert self.rank in [2,4], "SymmetricTensorDict can only handle rank 2 or 4 tensors."
        dict.__init__(self, *args, **kwargs)
        for (k,v) in self.items(): self[k]=v

        
    def __setitem__(self, key, value):
        if (self.rank==2):
            (a,b) = key
            dict.__setitem__(self,(a,b),value)
            dict.__setitem__(self,(b,a),value)
        else:
            for k in cov_syms(key): dict.__setitem__(self,k,value)

    def get_index_values(self):
        """
        Gets all the unique values which one of the indices can take.
        """
        if (self.rank==2): return set(i for p in self.keys() for i in p)
        else: return set(i for pp in self.keys() for p in pp for i in p)


class PowerSpectra():
    """
    Holds information about a set of powerspectra with some ell binning and possibly their covariance
    """
    spectra = cov = ells = None
    
    def __init__(self,*args,**kwargs):
        """
        PowerSpectra(**kwargs)
        PowerSpectra(spectra,cov,ells) where spectra/cov can be dicts or SymmetricTensorDicts
        PowerSpectra(spectra,cov,ells,maps) where spectra/cov are matrices
        """
        if (len(args))==1: self.spectra, = args
        elif (len(args))==3: self.spectra, self.cov, self.ells = args
        elif (len(args))==4: self.spectra, self.cov, self.ells, maps = args
        elif (len(args))==0: self.spectra, self.cov, self.ells, maps = (kwargs.pop(n,None) for n in ["spectra","cov","ells","maps"])
        else: raise TypeError("Expected 3 or 4 arguments or all named arguments")
              
        if (self.spectra==None): self.spectra = SymmetricTensorDict(rank=2)
        elif (isinstance(self.spectra,SymmetricTensorDict)): pass
        elif (isinstance(self.spectra,dict)): self.spectra = SymmetricTensorDict(self.spectra,rank=2)
        elif (isinstance(self.spectra,ndarray)):
            assert maps!=None, "You must provide a list of maps names."
            nps = len(pairs(maps))
            self.spectra = SymmetricTensorDict(zip(pairs(maps),partition(self.spectra,nps)),rank=2)
        else: raise ValueError("Expected spectra to be a vector or dictionary.")
            
        if (self.cov==None): self.cov = SymmetricTensorDict(rank=4)
        elif (isinstance(self.cov,SymmetricTensorDict)): pass
        elif (isinstance(self.cov,dict)): self.cov = SymmetricTensorDict(self.cov,rank=4)
        elif (isinstance(self.cov,ndarray)):
            assert maps!=None, "You must provide a list of maps names."
            nps = len(pairs(maps))
            nl = alen(self.cov)/nps
            self.cov = SymmetricTensorDict([((p1,p2),self.cov[i*nl:(i+1)*nl,j*nl:(j+1)*nl]) for (i,p1) in zip(range(nps),pairs(maps)) for (j,p2) in zip(range(nps),pairs(maps))],rank=4)
        else: raise ValueError("Expected covariance to be a matrix or dictionary.")
              
        assert self.spectra or self.ells!=None, "You must either provide some spectra or some ells"
        if self.ells==None: self.ells = arange(len(self.spectra.values()[0]))
            
            
    def __getitem__(self,key):
        return self.spectra[key]
    
    def __setitem__(self,key,value):
        self.spectra[key]=value
    
    def get_maps(self):
        return sorted(self.spectra.get_index_values())
    
    def get_as_matrix(self,ell_blocks=False):
        """
        Gets the spectra and covariance as matrices.
        
        If ell_blocks=True the matrices are block matrices where each blocks corresponds to the same ell
        otherwise (default) each block corresponds to the same powerspectra. 
        """
        cov_mat=None
        if ell_blocks:
            spec_mat = vstack([[[ell,self.spectra[k][iell]] for k in pairs(self.get_maps())] for (iell,ell) in enumerate(self.ells)])
            if (self.cov): 
                nl, nps = alen(self.ells), alen(pairs(self.get_maps()))
                cov_mat = zeros((nl*nps,nl*nps))
                for (i,p1) in enumerate(pairs(self.get_maps())):
                    for (j,p2) in enumerate(pairs(self.get_maps())):
                        cov_mat[i::nps,j::nps] = self.cov[(p1,p2)] if i>j else self.cov[(p1,p2)].T
        else:
            spec_mat = vstack(vstack([self.ells,self.spectra[(alpha,beta)]]).T for (alpha,beta) in pairs(self.get_maps()))
            if (self.cov): cov_mat = vstack(dstack(array([[(lambda c: c if (p1,p2) in pairs(pairs(self.get_maps())) else c.T)(self.cov[(p1,p2)]) for p1 in pairs(self.get_maps())] for p2 in pairs(self.get_maps())])))
            
        return namedtuple("SpecCov", ['spec','cov'])(spec_mat,cov_mat)

    def save_as_matrix(self,fileroot):
        spec_mat, cov_mat = self.get_as_matrix()
        savetxt(fileroot+"_spec",spec_mat,fmt="%8.2f %.5e")
        if cov_mat!=None: savetxt(fileroot+"_cov",cov_mat)    
    
    def save_as_files(self,fileroot):
        raise NotImplementedError
#        os.path.join(params["pcls"],"-".join(d1)+'__'+"-".join(d2)+'.dat'))
        
        
    def calibrated(self,maps,ells,weighting=1):
        """
        Calibrate some to powerspectra to eachother in some ell range with a given ell-weighting.
        Returns a dictionary of calibration factors which can be passed to PowerSpectra.lincombo
        """
        maps = list(maps)
        fid = mean([self.spectra[(a,b)][ells]*weighting for (a,b) in pairs(maps)],axis=0)
        def miscalib(calib):
            return sum(norm(self.spectra[(a,b)][ells]*weighting*calib[ia]*calib[ib] - fid) for ((a,ia),(b,ib)) in pairs(zip(maps,range(len(maps)))))
        calib = fmin(miscalib,ones(len(maps)))
        return [(newmap,[(newmap,calib[maps.index(newmap)] if newmap in maps else 1)]) for newmap in self.get_maps()]
        
    
    def lincombo(self,new_maps,normalize=True):
        """
        Recast a signal covariance into some new linear combinations of maps
        p is a PowerSpectra
        new_maps is a list of (freq,factor)'p where freq is a frequency in p.maps and factor is 
        the coefficient for that frequencies contribution.
        normalize indicates whether to normalize the linear combination so that sum(weights)=1
        
        Returns a new PowerSpectra structure
        """
        if isinstance(new_maps,dict): new_maps = new_maps.items()
        
        spectra = SymmetricTensorDict([
              ((alpha2,beta2),
               sum(self.spectra[(alpha,beta)]*fa*fb for (alpha,fa) in coeff_a for (beta,fb) in coeff_b) 
               /(sum(fa*fb for (_,fa) in coeff_a for (_,fb) in coeff_b) if normalize else 1))
              for ((alpha2,coeff_a),(beta2,coeff_b)) in pairs(new_maps)
          ],rank=2)
        
        if self.cov:
            cov = SymmetricTensorDict([
                  (((alpha2,beta2),(gamma2,delta2)),
                   sum(
                       self.cov[((alpha,beta),(gamma,delta))]*fa*fb*fg*fd
                       for (alpha,fa) in coeff_a for (beta,fb) in coeff_b for (gamma,fg) in coeff_g for (delta,fd) in coeff_d
                   )/(sum(
                       fa*fb*fg*fd
                       for (alpha,fa) in coeff_a for (beta,fb) in coeff_b for (gamma,fg) in coeff_g for (delta,fd) in coeff_d
                   ) if normalize else 1))
                  for (((alpha2,coeff_a),(beta2,coeff_b)),((gamma2,coeff_g),(delta2,coeff_d))) in pairs(pairs(new_maps))
              ],rank=4)
        else:
            cov = None
        
        return PowerSpectra(spectra,cov,self.ells)
    
    def plot(self,which=None,errorbars=True,prefix="",**kwargs):
        if (which==None): which = pairs(self.get_maps())
        if errorbars and self.cov:
            for (a,b) in which:
                errorbar(self.ells,self.spectra[(a,b)],yerr=sqrt(diagonal(self.cov[((a,b),(a,b))])),
                         label=prefix+str(a)+" x "+str(b),fmt=kwargs.pop("fmt","."),**kwargs)
        else:
            for (a,b) in which:
                plot(self.ells,self.spectra[(a,b)],label=str(a)+" x "+str(b),**kwargs)
        yscale("log")


    def apply_func(self,fspec,fcov):
        spectra = SymmetricTensorDict([(k,fspec(k,self.spectra[k])) for k in pairs(self.get_maps())],rank=2)
        if (self.cov): cov = SymmetricTensorDict([(k,fcov(k,self.cov[k])) for k in pairs(pairs(self.get_maps()))],rank=4)
        else: cov = None
        return PowerSpectra(spectra,cov,self.ells)

    def to_dl(self):
        return self.rescaled(self.ells**2/(2*pi))

    def to_cl(self):
        return self.rescaled(2*pi/self.ells**2)

    def rescaled(self,fac):
        return self.apply_func(lambda _, spec: spec*fac, lambda _, cov: cov*outer(fac,fac))
    
    def binned(self,bin):
        ps = self.apply_func(lambda _, spec: bin(spec), lambda _, cov: bin(cov))
        ps.ells = bin(self.ells)
        ps.bin = bin
        return ps
    
    def sliced(self,*args):
        if len(args)==1 and type(args[0])==slice: s=args[0]
        else: s=slice(*args)
        ps = self.apply_func(lambda _, spec: spec[s], lambda _, cov: cov[s,s])
        ps.ells = self.ells[s]
        return ps
    
    def __add__(self,other):
        return PowerSpectra.sum([self,other])


    def __sub__(self,other):
        assert all(self.ells==other.ells), "Can't add spectra with different ells"
        maps = list(set(self.get_maps()) & set(other.get_maps()))
        spectra = SymmetricTensorDict([(k,self.spectra[k]-other.spectra[k]) for k in pairs(maps)],rank=2)
        #TODO: Add covariances
        return PowerSpectra(spectra,None,self.ells)
        
        
    @staticmethod
    def sum(ps):
        ells = [p.ells for p in ps]
        assert all(ell == ells[0] for ell in ells), "Can't add spectra with different ells."
        maps = reduce(lambda x, y: x & y, [set(p.get_maps()) for p in ps])
        spectra = SymmetricTensorDict([(k,sum(p.spectra[k] for p in ps)) for k in pairs(maps)],rank=2)
        #TODO: Add covariances
        return PowerSpectra(spectra,None,ells[0])


def read_AP_ini(params):
    """
    Read and process the americanpypeline ini file
    """
    if type(params)==str:
        p = read_ini(params)
        p["lmin"]=int(p.get("lmin",0))
        p["lmax"]=int(p["lmax"])
        p.update(get_mask_info(p["mask"]))
        if "cleaning" in p: p["cleaning"]=[(outmap,[(str(inmap),coeff) for (inmap,coeff) in lc]) for (outmap,lc) in literal_eval(p["cleaning"]).items()]
        else: p["cleaning"]=None
        p["pcl_binning"]=get_bin_func(p.get("pcl_binning","none"))
        p["mcmc_binning"]=get_bin_func(p.get("mcmc_binning","none"))
        return p
    else:
        return params



def alm2cl(alm1,alm2):
    """
    Compute the cross power-spectrum given two alms computed with map2alm
    Assumes these are alm's from a real map so that a_lm = -conjugate(a_l(-m))
    """
    (lmax1,lmax2) = [H.Alm.getlmax(alen(a)) for a in [alm1,alm2]]
    lmax = min(lmax1,lmax2)
    return array(real([(alm1[l]*alm2[l]+2*real(vdot(alm1[H.Alm.getidx(lmax1,l,arange(1,l+1))],alm2[H.Alm.getidx(lmax2,l,arange(1,l+1))])))/(2*l+1) for l in range(lmax)]))



def skycut_mask(nside,percent,dth=deg2rad(10),taper_fn=lambda x: cos(pi*(x-1)/4)**2):
    """
    Returns an NSIDE ring-scheme healpix mask with percent of the sphere masked out around the galactic plane.
    
    dth and taper_fn describe the angular extent and apodizing function with which to taper the transition.
    The default is a Hann window over 10 degrees
    """
    mask = ones(12*nside**2)
    thcut = arcsin(percent)
    for iz in range(4*nside+1):
        th = arcsin(H.ring2z(nside,iz))
        mask[H.in_ring(nside,iz,0,pi)] = 0 if abs(th)<thcut-dth/2 else 1 if abs(th)>thcut+dth/2 else taper_fn((abs(th)-thcut)/dth)

    return mask


def get_mask_info(maskstr):
    """
    Given the mask line in the parameter file like '70% + ptsrc.fits'
    returns a dict with the names of the .mll, .imll, .mll2, and .imll2 files 
    and the percentage
    """
    
    (percent,maskfile) = re.match("(?:([0-9]+)%\s*\+\s*)?(.*)",maskstr).groups()
    if percent==None: percent=0
    
    ret = dict([("mask_"+post,maskfile+"."+percent+"."+post) for post in ["mll","imll","mll2","imll2"]])
    ret["mask_percent"]=percent
    ret["mask_file"]=maskfile
    
    return ret
    
    
def get_mask(maskstr,Nside=None):
    """
    Given the mask line in the parameter file like '70% + ptsrc.fits'
    returns the healpix mask.
    
    If Nside is present, upgrades/degrades the map to the given Nside
    """
    params = get_mask_info(maskstr)
    mask = H.read_map(params["mask_file"])
    if (Nside==None): Nside = H.npix2nside(alen(mask))
    else: mask = H.ud_grade(mask,Nside)
    if params["mask_percent"]!=0: mask*=skycut_mask(Nside, float(params["mask_percent"])/100.)
    return mask


def get_bin_func(binstr):
    """
    Returns a binning function bin(x) where x can be either a vector, matrix, or slice object. 
    """
    binstr = binstr.lower()
    
    if (binstr=="none"): return lambda x: x
    
    if (binstr=='wmap'): 
        wmap=loadtxt(os.path.join(AProotdir,"dat/external/wmap_binned_tt_spectrum_7yr_v4p1.txt"))
        wmapbins=[slice(s,e+1) for [s,e] in wmap[:-2,[1,2]]]+[slice(l,l+50) for l in range(1001,3000,50)]
        def wmapbin(cl):
            if type(cl)==slice: raise NotImplementedError("Can't slice WMAP bins yet")
            else:
                bins = wmapbins[:bisect_right([s.start for s in wmapbins],len(cl))-1]
                binned = array([mean(cl[s],axis=0) for s in bins])
                return binned if len(shape(cl))==1 else array(map(lambda cl: wmapbin(cl),binned))
        return wmapbin
    
    r = re.match("flat\((?:dl=)?([0-9]+)\)",binstr)
    if (r!=None):
        dl=int(r.group(1))
        def flatbin(cl):
            if type(cl)==slice: return slice(cl.start/dl if cl.start else None,cl.stop/dl,cl.step/dl if cl.step else None)
            else:
                b = array([mean(cl[l:l+dl],axis=0) for l in dl*arange(alen(cl)/dl)])
                return b if len(shape(cl))==1 else array(map(lambda cl: flatbin(cl),b))
        return flatbin
    
    raise ValueError("Unknown binning function '"+binstr+"'")


def load_signal(params, clean=False, calib=False, calibrange=slice(150,300)):
    params = read_AP_ini(params)
    ells = arange(params["lmax"])
    spectra = load_multi(params["signal"]+"_spec")[:,1]
    try: cov = load_multi(params["signal"]+"_cov")
    except IOError: cov = None
    signal = PowerSpectra(spectra, cov, ells, freqs)
    if clean and params["cleaning"]!=None: signal = signal.lincombo(params["cleaning"])
    if calib: signal = signal.lincombo(signal.calibrated(signal.get_maps(),params["pcl_binning"](calibrange)),normalize=False)
    return signal


def load_clean_calib_signal(params,calibrange=slice(150,300)):
    return load_signal(params, True, True, calibrange)


def load_pcls(params):
    params = read_AP_ini(params)
    bin = params["pcl_binning"]
    lmax = params["lmax"]
    ells = bin(arange(lmax))
    spectra = SymmetricTensorDict(rank=2)
    for f in os.listdir(params["pcls"]):
        (a,b) = tuple(MapID(*s.split("-")) for s in f.replace(".dat","").split("__"))
        pcl = loadtxt(os.path.join(params["pcls"],f))[:lmax]
        assert alen(pcl)>=lmax, "Pseudo-cl's have not been calculated to high enough lmax. Please run maps_to_pcls.py again." 
        spectra[(a,b)] = bin(pcl)
    return PowerSpectra(spectra,None,ells)



def load_beams(params):
    params = read_AP_ini(params)
    bin = params["pcl_binning"]
    regex=re.compile("(100|143|217|353).*?([1-8][abc]?)")
    files = [(os.path.join(params["beams"],f),regex.search(f)) for f in os.listdir(params["beams"])]
    return dict([(MapID(r.group(1),'T',r.group(2)),loadtxt(f)[:int(params["lmax"]),1]) for (f,r) in files if r!=None])
    
    
def cmb_orient(wmap=True,spt=True):
    if (wmap):
        wmap = loadtxt(os.path.join(AProotdir,"dat/external/wmap_binned_tt_spectrum_7yr_v4p1.txt"))
        errorbar(wmap[:,0],wmap[:,3],yerr=wmap[:,5],fmt='.',label='WMAP7')
    if (spt):
        spt = loadtxt(os.path.join(AProotdir,"dat/external/dl_spt20082009.txt"))
        errorbar(spt[:,0],spt[:,1],yerr=spt[:,2],fmt='.',label='SPT K11')

