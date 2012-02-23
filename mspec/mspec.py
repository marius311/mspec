from ast import literal_eval
from bisect import bisect_right, bisect_left
from collections import namedtuple
from matplotlib.pyplot import plot, errorbar, contour, yscale as ppyscale, xscale as ppxscale
from matplotlib.mlab import movavg
from numpy import *
from numpy.linalg import norm
from scipy.optimize import fmin
from utils import *
import healpy as H
import sys, os, re, gc

MapID = namedtuple("MapID", ["fr","type","id"])
MapID.__str__ = MapID.__repr__ = lambda self: "-".join(self)  

def_map_regex = "(100|143|217|353).*?([0-9][abc]?)"


class SymmetricTensorDict(dict):
    """
    A dictionary where keys are either (a,b) or ((a,b),(c,d)). 
    When a key is added, other keys which have the same value because of symmetry are also added.
    
    If created with rank=2 the symmetry properties are 
        (a,b) = (b,a)
        
    If created with rank=4 the symmetry properties are 
        ((a,b),(c,d)) = ((b,a),(c,d)) = ((a,b),(d,c)) = ((c,d),(a,b))^T
    """
    
    def __init__(self, *args, **kwargs):
        """
        If created with rank=2 the symmetry properties are 
            (a,b) = (b,a)
        
        If created with rank=4 the symmetry properties are 
            ((a,b),(c,d)) = ((b,a),(c,d)) = ((a,b),(d,c)) = ((c,d),(a,b))^T
        """
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
            ((a,b),(c,d)) = key
            for k in set([((a,b),(c,d)),((b,a),(c,d)),((a,b),(d,c)),((b,a),(d,c))]): dict.__setitem__(self,k,value)
            for k in set([((c,d),(a,b)),((c,d),(b,a)),((d,c),(a,b)),((d,c),(b,a))]): dict.__setitem__(self,k,value.T if type(value)==ndarray else value)


    def get_index_values(self):
        """Gets all the unique values which one of the indices can take."""
        if (self.rank==2): return set(i for p in self.keys() for i in p)
        else: return set(i for pp in self.keys() for p in pp for i in p)


class PowerSpectra():
    """
    Holds information about a set of powerspectra and possibly their covariance.
    
    Powerspectra are stored in the .spectra field and can also
    be accessed directly by indexing this object, e.g.
        >> p[('143','217')] = ...
        
    Covariances are stored in the .cov field and can be accessed by,
        >> p.cov[(('143','217'),('143','217'))] = ...
        
    Spectra and covariances are automatically symmetrized, so when one key is added
    other keys which have the same value because of symmetry are also added. For example,
        >> p[('143','217')] = x
        >> p[('143','217')] == p[('217','143')]
        True
    
    All of the functions on this object automatically propagate uncertainties in the covariance.    
    """
    spectra = cov = ells = None
    
    def __init__(self,spectra=None,cov=None,ells=None,maps=None):
        """
        Create a PowerSpectra instance.
        
        Keyword arguments:
            spectra/cov -- Dicts or SymmetricTensorDicts or block vectors/matrices (for example
                           the output of PowerSpectra.get_as_matrix()). If they are block vectors/matrices
                           then maps must be provided. 
            ells -- An array of ell values which must be the same length as spectra
                    (default=arange(lmax) where lmax is determined from the length of the spectra)
            maps -- If spectra and/or cov are provided as block matrices, a list of maps names
                    must provided to partition the matrices and name the blocks. 
        """
        if (spectra==None): self.spectra = SymmetricTensorDict(rank=2)
        elif (isinstance(spectra,SymmetricTensorDict)): self.spectra = spectra
        elif (isinstance(spectra,dict)): self.spectra = SymmetricTensorDict(spectra,rank=2)
        elif (isinstance(spectra,ndarray)):
            assert maps!=None, "You must provide a list of maps names."
            nps = len(pairs(maps))
            self.spectra = SymmetricTensorDict(zip(pairs(maps),partition(spectra,nps)),rank=2)
        else: raise ValueError("Expected spectra to be a vector or dictionary.")
            
        if (self.cov==None): self.cov = SymmetricTensorDict(rank=4)
        elif (isinstance(cov,SymmetricTensorDict)): self.cov = cov
        elif (isinstance(cov,dict)): self.cov = SymmetricTensorDict(cov,rank=4)
        elif (isinstance(self.cov,ndarray)):
            assert maps!=None, "You must provide a list of maps names."
            nps = len(pairs(maps))
            nl = alen(cov)/nps
            self.cov = SymmetricTensorDict([((p1,p2),cov[i*nl:(i+1)*nl,j*nl:(j+1)*nl]) for (i,p1) in zip(range(nps),pairs(maps)) for (j,p2) in zip(range(nps),pairs(maps)) if i<=j],rank=4)
        else: raise ValueError("Expected covariance to be a matrix or dictionary.")
              
        assert self.spectra or self.ells!=None, "You must either provide some spectra or some ells"
        if self.ells==None: self.ells = arange(len(self.spectra.values()[0]))
            
    def __getitem__(self,key):
        return self.spectra[key]
    
    def __setitem__(self,key,value):
        self.spectra[key]=value
    
    def get_maps(self):
        """Gets keys for all maps"""
        return sorted(self.spectra.get_index_values())

    def get_spectra(self):
        """Gets keys for all spectra"""
        return pairs(self.get_maps())

    def get_auto_spectra(self):
        """Gets keys for all auto spectra"""
        return [(m,m) for m in self.get_maps()]

    def get_cross_spectra(self):
        """Gets keys for all cross spectra"""
        return set(self.get_spectra()) - set(self.get_auto_spectra())

    def get_as_matrix(self,ell_blocks=False):
        """
        Gets the spectra as a single block vector and the covariances as
        a single block matrix. The return value has fields 'spec' and 'cov. 
        
        Keyword arguments:
        ell_blocks -- If True each block corresponds to the same ell, otherwise each 
                      block corresponds to the same spectrum. (default=False)
        """
        cov_mat=None
        if ell_blocks:
            spec_mat = vstack([[[ell,self.spectra[k][iell]] for k in pairs(self.get_maps())] for (iell,ell) in enumerate(self.ells)])
            if (self.cov): 
                nl, nps = alen(self.ells), alen(pairs(self.get_maps()))
                cov_mat = zeros((nl*nps,nl*nps))
                for (i,p1) in enumerate(pairs(self.get_maps())):
                    for (j,p2) in enumerate(pairs(self.get_maps())):
                        cov_mat[i::nps,j::nps] = self.cov[(p1,p2)]
        else:
            spec_mat = vstack(vstack([self.ells,self.spectra[(alpha,beta)]]).T for (alpha,beta) in pairs(self.get_maps()))
            if (self.cov): cov_mat = vstack(dstack(array([[self.cov[(p1,p2)] for p1 in pairs(self.get_maps())] for p2 in pairs(self.get_maps())])))
            
        return namedtuple("SpecCov", ['spec','cov'])(spec_mat,cov_mat)

    def save_as_matrix(self,fileroot):
        """Save the matrix representation of this PowerSpectra to file"""
        spec_mat, cov_mat = self.get_as_matrix()
        save_multi(fileroot+"_spec",spec_mat)
        if cov_mat!=None: save_multi(fileroot+"_cov",cov_mat)    
    
    def calibrated(self,maps,ells,weighting=1):
        """
        Return calibration factors corresponding to calibrating 
        some maps to their mean. The output of this function can then be passed to 
        PowerSpectra.lincombo
        
        Keyword arguments:
        maps -- Which maps to calibrate
        ells -- An array or a slice object corresponding to the ell range in which
                to do the calibration.
        weighting -- The weighting to use when calculating the 'miscalibration'. For example
                     is the spectra are C_ell's, then weighting = 2*ells+1 corresponds to
                     map-level calibration. 
        """
        maps = list(maps)
        fid = mean([self.spectra[(a,b)][ells]*weighting for (a,b) in pairs(maps)],axis=0)
        def miscalib(calib):
            return sum(norm(self.spectra[(a,b)][ells]*weighting*calib[ia]*calib[ib] - fid) for ((a,ia),(b,ib)) in pairs(zip(maps,range(len(maps)))))
        calib = fmin(miscalib,ones(len(maps)),disp=True)
        return [(newmap,[(newmap,calib[maps.index(newmap)] if newmap in maps else 1)]) for newmap in self.get_maps()]
        
    
    def lincombo(self,new_maps,normalize=True):
        """
        Return a new PowerSpectra corresponding to forming a linear combination of maps.
        
        Keyword arguments:
        new_maps -- A dictionary mapping the new maps to linear combinations of the old maps.
                    For example, {'217c': [('217',1),('353',-.14)], '143c': [('143',1),('353',-.038)]}
        normalize -- Whether to normalize the sum of the weights to 1 (i.e. keep the CMB constant)
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
        """
        Plot these powerspectra.
        
        Keyword arguments:
        which -- A list of keys to the spectra which should be plotted (default=all of them)
        errorbars -- Whether to plot error bars (default=True)
        prefix -- A prefix which shows up in the legend (default="")
        
        Other arguments:
        **kwargs -- These are passed through to the plot function.
        """
        if (which==None): which = pairs(self.get_maps())
        if errorbars and self.cov:
            for (a,b) in which:
                errorbar(self.ells,self.spectra[(a,b)],yerr=sqrt(diagonal(self.cov[((a,b),(a,b))])),
                         label=prefix+str(a)+" x "+str(b),fmt=kwargs.pop("fmt","."),**kwargs)
        else:
            for (a,b) in which:
                plot(self.ells,self.spectra[(a,b)],label=prefix+str(a)+" x "+str(b),**kwargs)


    def apply_func(self,fspec=lambda _,x: x, fcov=lambda _, x: x):
        """
        Apply an arbitrary function to each spectrum and covariance.
        
        Keyword arguments:
        fspec -- A function of (k,s) applied to the spectra where k is the 
                 powerspectrum key and s is the spectrum (default=identity)
        fspec -- A function of (k,c) applied to the covariances where k is the 
                 covariance key and c is the covariance (default=identity) 
        """
        spectra = SymmetricTensorDict([(k,fspec(k,self.spectra[k])) for k in pairs(self.get_maps())],rank=2)
        if self.cov: cov = SymmetricTensorDict([(k,fcov(k,self.cov[k])) for k in pairs(pairs(self.get_maps()))],rank=4)
        else: cov = None
        return PowerSpectra(spectra,cov,self.ells)

    def dl(self):
        """Multilies all spectra by ell(ell+1)/2/pi"""
        return self.rescaled(self.ells**2/(2*pi))

    def cl(self):
        """Divides all spectra by ell(ell+1)/2/pi"""
        return self.rescaled(2*pi/self.ells**2)

    def rescaled(self,fac):
        """Multiplies all spectra by a constant or ell-dependent factor"""
        return self.apply_func(lambda _, spec: spec*fac, lambda _, cov: cov*outer(fac,fac))
    
    def binned(self,bin):
        """ 
        Returns a binned version of this PowerSpectra. 
        
        Keyword arguments:
        bin -- Either a binning function which takes as input both vectors and matrices, 
               or a string which is passed to get_bin_func
        """
        if (type(bin)==str): bin = get_bin_func(bin)
        ps = self.apply_func(lambda _, spec: bin(spec), lambda _, cov: bin(cov))
        ps.ells = bin(self.ells)
        ps.bin = bin
        return ps
    
    def sliced(self,*args):
        """
        Slices each spectra. The argument can be a single slice object, or 
        1, 2, or 3 arguments which are passed to slice.
        """
        if len(args)==1 and type(args[0])==slice: s=args[0]
        else: s=slice(*args)
        ps = self.apply_func(lambda _, spec: spec[s], lambda _, cov: cov[s,s])
        ps.ells = self.ells[s]
        return ps
    
    def diffed(self,fid):
        """Differences each spectra against a fiducial template"""
        assert alen(fid)==alen(self.ells), "Must difference against power-spectrum with same number of ells"
        return self.apply_func(lambda _, cl: cl-fid, lambda _,cov: cov)
        
    def __add__(self,other):
        return PowerSpectra.sum([self,other])

    def __sub__(self,other):
        return PowerSpectra.sum([self,other.rescaled(-1)])
        
    @staticmethod
    def sum(ps):
        """Return the sum of one or more PowerSpectra."""
        ells = [p.ells for p in ps]
        assert all(ell == ells[0] for ell in ells), "Can't add spectra with different ells."
        maps = reduce(lambda x, y: x & y, [set(p.get_maps()) for p in ps])
        spectra = SymmetricTensorDict([(k,sum(p.spectra[k] for p in ps)) for k in pairs(maps)],rank=2)
        if all([p.cov for p in ps]): cov = SymmetricTensorDict([(k,sum(p.cov[k] for p in ps)) for k in pairs(pairs(maps))],rank=4)
        else: cov=None
        return PowerSpectra(spectra,cov,ells[0])



def read_Mspec_ini(params,relative_paths=True):
    """
    Read and process an mspec ini file, returning a dictionary.
    
    Keyword arguments:
    params -- A string filename, or a dictionary for an already loaded file.
    relative_path -- If true and params is a filename, all relative paths in the file 
                     are turned into absolute paths relative to the parameter file 
                     itself. (default=True)
    """
    if type(params)==str:
        p = read_ini(params)
        
        for (k,v) in p.items():
            if (type(v)==str):
                try: 
                    pv = literal_eval(v)
                    if isinstance(pv, list): pv = array(pv)
                except: pv = try_type(v)
            if relative_paths and type(pv)==str:
                rel = os.path.abspath(os.path.join(os.path.dirname(params),pv))
                if os.path.exists(rel): pv=rel
            p[k]=pv
            
        p["binning"]=get_bin_func(p.get("binning","none"))
        
    return p



def alm2cl(alm1,alm2):
    """
    Compute the cross power-spectrum given two alms computed with map2alm
    Assumes these are alm's from a real map so that a_lm = -conjugate(a_l(-m))
    """
    (lmax1,lmax2) = [H.Alm.getlmax(alen(a)) for a in [alm1,alm2]]
    lmax = min(lmax1,lmax2)
    return array(real([(alm1[l]*alm2[l]+2*real(vdot(alm1[H.Alm.getidx(lmax1,l,arange(1,l+1))],alm2[H.Alm.getidx(lmax2,l,arange(1,l+1))])))/(2*l+1) for l in range(lmax)]))

def skycut_mask(nside,percent,dth=deg2rad(10),apod_fn=lambda x: cos(pi*(x-1)/4)**2):
    """
    Returns an ring-scheme healpix mask with a given percent of the sphere 
    masked out around the galactic plane.
    
    Keyword arguments:
    
    nside -- The nside of the mask
    percent -- How much of the sky to mask
    dth -- The number of radians of apodization (default = 10 degrees)
    apod_fn -- The apodization function (default = Hann window)
    """
    mask = ones(12*nside**2)
    thcut = arcsin(percent)
    for iz in range(4*nside+1):
        th = arcsin(H.ring2z(nside,iz))
        mask[H.in_ring(nside,iz,0,pi)] = 0 if abs(th)<thcut-dth/2 else 1 if abs(th)>thcut+dth/2 else apod_fn((abs(th)-thcut)/dth)

    return mask


def get_bin_func(binstr):
    """
    Returns a binning function bin(x,axis=None) where x can be any rank array and 
    axis specifies the binning axis, or None for all axes.
    
    Valid binning functions are:
    WMAP -- WMAP binning at low-ell, SPT binning at high-ell
    CTP -- CTP binning
    flat(x) -- Uniform bins of width x
    """
    binstr = binstr.lower()
    
    if (binstr=="none"): return lambda x: x
    
    bindat = None
    
    if (binstr=="ctp"):
        ctpbins=loadtxt(os.path.join(Mrootdir,"dat/bins/CTP_bin_TT_orig"),dtype=int)
        bindat=[arange(s,e+1) for [s,e] in ctpbins[:,[1,2]]]
    
    if (binstr=='wmap'): 
            wmapbins=loadtxt(os.path.join(Mrootdir,"dat/external/wmap_binned_tt_spectrum_7yr_v4p1.txt"),dtype=int)
            bindat=[arange(s,e+1) for [s,e] in wmapbins[:-2,[1,2]]]+[arange(l,l+50) for l in range(1001,2000,50)]+[arange(l,l+200) for l in range(2001,4000,200)]

    r = re.match("flat\((?:dl=)?([0-9]+)\)",binstr)
    if (r!=None):
        dl=int(r.group(1))
        bindat=[arange(l,l+dl) for l in dl*arange(10000/dl)]
    
    def bin(x,axis=None):
        if type(x)==slice:
            return slice(bisect_left([s[0] for s in bindat],x.start),bisect_left([s[-1] for s in bindat],x.stop))
        else:
            bins = bindat[:bisect_left([s[-1] for s in bindat],len(x))]
            for a in ([axis] if axis!=None else range(x.ndim)): x = array([x.take(s,axis=a).mean(axis=a) for s in bins]).swapaxes(a,0)
            return x

    if bindat!=None: return bin
    else: raise ValueError("Unknown binning function '"+binstr+"'")


def load_signal(params, clean=False, calib=False, calibrange=slice(150,500), loadcov=True):
    """
    Loads the signal computed by pcl_to_signal.py
    
    Keyword arguments:
    params -- The parameter file
    clean -- Whether to apply cleaning (default=False)
    calib --Whether to do inter-frequency calibration (default=False)
    calibrange -- A slice object corresponding to the range of ells over which
                  to do the calibration (default=slice(150,500))
    loadcov -- Whether to load the covariance (default=True)
    """
    params = read_Mspec_ini(params)
    bin = params['binning']
    ells, spectra = load_multi(params["signal"]+"_spec").T
    ells = array(sorted(set(ells)))
    cov = None
    if loadcov and str2bool(params.get("get_covariance",False)):
        try: cov = load_multi(params["signal"]+"_cov")
        except IOError: pass
    signal = PowerSpectra(spectra, cov, ells, params["freqs"])
    if clean and "cleaning" in params: signal = signal.lincombo(params["cleaning"])
    if calib: signal = signal.lincombo(signal.calibrated(signal.get_maps(),calibrange),normalize=False)
    return signal


def load_clean_calib_signal(params, calibrange=slice(150,500), loadcov=True):
    """
    Loads the cleaned and calibrated signal computed by pcl_to_signal.py
    
    Keyword arguments:
    params -- The parameter file
    calibrange -- A slice object corresponding to the range of ells over which
                  to do the calibration (default=slice(150,500))
    loadcov -- Whether to load the covariance (default=True)
    """

    return load_signal(params, True, True, calibrange, loadcov)


def load_pcls(params):
    """Load the pcls computed by maps_to_pcls.py"""
    params = read_Mspec_ini(params)
    regex=re.compile(params.get("map_regex",def_map_regex))
    files = [(os.path.join(params["pcls"],f),[MapID(m.group(1),'T',m.group(2)) for m in regex.finditer(f)]) for f in os.listdir(params["pcls"])]
    return PowerSpectra({tuple(m):load_multi(f)[:int(params["lmax"])] for (f,m) in files if len(m)==2})


def load_beams(params):
    """Load the beams"""
    params = read_Mspec_ini(params)
    regex=re.compile(params.get("map_regex",def_map_regex))
    files = [(os.path.join(params["beams"],f),[MapID(m.group(1),'T',m.group(2)) for m in regex.finditer(f)]) for f in os.listdir(params["beams"])]
    beams = SymmetricTensorDict(rank=2)
    for (f,m) in files:
        if len(m)==1: beams[(m[0],m[0])] = load_multi(f)[:int(params["lmax"]),params.get("beam_col",1)]**2
        elif len(m)==2: beams[(m[0],m[1])] = load_multi(f)[:int(params["lmax"]),params.get("beam_col",1)]**2
        
    for (m1,m2) in pairs(beams.get_index_values()):
        if (m1,m2) not in beams: 
            beams[(m1,m2)] = sqrt(beams[(m1,m1)]*beams[(m2,m2)])
    
    return PowerSpectra(beams)
        
def load_noise(params):
    """Load the noise"""
    params = read_Mspec_ini(params)
    regex=re.compile(params.get("map_regex",def_map_regex))
    files = [(os.path.join(params["noise"],f),regex.search(f)) for f in os.listdir(params["noise"])]
    return dict([(MapID(r.group(1),'T',r.group(2)),load_multi(f)[:int(params["lmax"]),params.get("noise_col",1)]) for (f,r) in files if r!=None])
        
def cmb_orient(wmap=True,spt=True):
    """Plot up WMAP and SPT data points"""
    if (wmap):
        wmap = loadtxt(os.path.join(Mrootdir,"dat/external/wmap_binned_tt_spectrum_7yr_v4p1.txt"))
        errorbar(wmap[:,0],wmap[:,3],yerr=wmap[:,4],fmt='.',label='WMAP7')
    if (spt):
        spt = loadtxt(os.path.join(Mrootdir,"dat/external/dl_spt20082009.txt"))
        errorbar(spt[:,0],spt[:,1],yerr=spt[:,2],fmt='.',label='SPT K11')



