from ast import literal_eval
from bisect import bisect_right, bisect_left
from collections import namedtuple, defaultdict, OrderedDict
from itertools import takewhile, chain
from numpy import *
from numpy.linalg import norm
from scipy.optimize import fmin
from utils import *
import sys, os, re, gc
import os.path as osp


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

    def setdefault(self, key, value):
        if key not in self: self[key]=value
        return self[key]

    def get_index_values(self):
        """Gets all the unique values which one of the indices can take."""
        if (self.rank==2): return set(i for p in self.keys() for i in p)
        else: return set(i for pp in self.keys() for p in pp for i in p)


class PowerSpectra(object):
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
        >> p[('217','143')] == x
        True
    
    All of the functions on this object automatically propagate uncertainties in the covariance.    
    """
    spectra = cov = ells = binning = None
    
    def __init__(self,spectra=None,cov=None,ells=None,maps=None,binning=None):
        """
        Create a PowerSpectra instance.
        
        Keyword arguments:
            spectra/cov -- Dicts or SymmetricTensorDicts or block vectors/matrices 
                           (for example the output of PowerSpectra.get_as_matrix()).
                           If they are block vectors/matrices then maps 
                           must be provided. 
            ells -- An array of ell values which must be the same length as 
                    spectra (default=arange(lmax) where lmax is determined from 
                    the length of the spectra)
            maps -- If spectra and/or cov are provided as block matrices, a list
                    of maps names must provided to partition the matrices and 
                    name the blocks. 
            binning -- Specifies the binning used for these powerspecta. 
        """
        if (spectra==None): self.spectra = SymmetricTensorDict(rank=2)
        elif (isinstance(spectra,SymmetricTensorDict)): self.spectra = spectra
        elif (isinstance(spectra,dict)): self.spectra = SymmetricTensorDict(spectra,rank=2)
        elif (isinstance(spectra,ndarray)):
            assert maps!=None, "You must provide a list of maps names."
            nps = len(pairs(maps))
            self.spectra = SymmetricTensorDict(zip(pairs(maps),partition(spectra,nps)),rank=2)
        else: raise ValueError("Expected spectra to be a vector or dictionary.")
            
        if (cov==None): self.cov = SymmetricTensorDict(rank=4)
        elif (isinstance(cov,SymmetricTensorDict)): self.cov = cov
        elif (isinstance(cov,dict)): self.cov = SymmetricTensorDict(cov,rank=4)
        elif (isinstance(cov,ndarray)):
            assert maps!=None, "You must provide a list of maps names."
            nps = len(pairs(maps))
            nl = alen(cov)/nps
            self.cov = SymmetricTensorDict([((p1,p2),cov[i*nl:(i+1)*nl,j*nl:(j+1)*nl]) for (i,p1) in zip(range(nps),pairs(maps)) for (j,p2) in zip(range(nps),pairs(maps)) if i<=j],rank=4)
        else: raise ValueError("Expected covariance to be a matrix or dictionary.")
        if (ells!=None): self.ells=ells        
        self.binning = binning
        
        assert self.spectra or self.ells!=None, "You must either provide some spectra or some ells"
        if self.ells==None: 
            bmax = max(s.shape[0] for s in self.spectra.values())
            if self.binning is None: self.ells = arange(bmax)
            else: self.ells = self.binning(arange(20000))[:bmax]
            
    def __getitem__(self,key):
        if not isinstance(key,tuple): key = (key,key)
        return self.spectra[key]
    
    def __setitem__(self,key,value):
        self.spectra[key]=value
    
    def deepcopy(self):
        return PowerSpectra(ells=self.ells.copy(),
                     spectra={k:v.copy() for k,v in self.spectra.items()},
                     cov={k:v.copy() for k,v in self.cov.items()} if self.cov else None,
                     binning=self.binning)


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

    def get_as_matrix(self, lrange, ell_blocks=False, get_cov=True):
        """
        Gets the spectra as a single block vector and the covariances as
        a single block matrix. The return value has fields 'spec' and 'cov. 
        
        Keyword arguments:
        lrange -- Dictionary mapping power spectra to ranges of ell 
                      which to include in the matrix. 
        ell_blocks -- If True each block corresponds to the same ell, otherwise each 
                      block corresponds to the same spectrum. (default=False)
        """
        cov_mat=None
#        if lrange is None: lrange = {}
#        for k in self.get_spectra(): lrange.setdefault(k,[None])
        lrange = OrderedDict([(k,self.binning(slice(*v))) for k,v in lrange.items()])            
        if ell_blocks:
            raise NotImplementedError("Functionality temporarily removed.")
        else:
            spec_mat = hstack(self.spectra[k][v] for k,v in lrange.items())
            if (get_cov and self.cov): 
                cov_mat = vstack(map(hstack,[[self.cov[(k1,k2)][v1,v2] 
                                              for k2,v2 in lrange.items()] for k1,v1 in lrange.items()]))
        return namedtuple("SpecCov", ['spec','cov'])(spec_mat,cov_mat)



    def save_as_matrix(self,fileroot):
        """Save the matrix representation of this PowerSpectra to file"""
        spec_mat, cov_mat = self.get_as_matrix()
        save_multi(fileroot+"_spec",spec_mat,npy=False)
        if cov_mat!=None: save_multi(fileroot+"_cov",cov_mat)    
    
    def save_as_fits(self, fileroot):
        from pyfits import new_table, ColDefs, Column, HDUList, PrimaryHDU, Header
        hdus = []
        spectra = new_table(ColDefs([Column(name=str(ps), format='D', array=self[ps]) for ps in pairs(self.get_maps()) if ps in self.spectra]))
        spectra.name = 'SPECTRA'
        hdus.append(spectra)
        
        if self.cov is not None:
            cov = new_table(ColDefs([Column(name=str(ps), format='D', array=self.cov[ps].ravel()) for ps in pairs(pairs(self.get_maps()))]))
            cov.name = 'COVS'
            flatcov = new_table(ColDefs([Column(name='cov', format='D', array=self.get_as_matrix().cov.ravel())]))
            flatcov.name = 'FULLCOV'
            hdus.extend([cov,flatcov])

        if self.binning is not None:
            nq = max([len(x) for x in self.spectra.values()])
            binning = new_table(ColDefs([Column(name='bins', format='D', array=self.binning.q[:nq].ravel())]))
            binning.name = 'BINNING'
            hdus.append(binning)            
            
        HDUList([PrimaryHDU()]+hdus).writeto(fileroot,clobber=True)
        
    @staticmethod
    def load_from_fits(fileroot):
        from pyfits import open
        
        spectra, cov, binning, ells = [None]*4
        
        with open(fileroot) as f:
            exts = {ext.name:ext for ext in f}
            
            spectra = {eval(n):exts['SPECTRA'].data[n] for n in exts['SPECTRA'].columns.names}
            nq = len(spectra.values()[0])
               
            if 'COVS' in exts:
                cov = {eval(n):exts['COVS'].data[n].reshape(nq,nq) for n in exts['COVS'].columns.names}
                
            if 'BINNING' in exts:
                q = exts['BINNING'].data['bins']
                q = q.reshape((nq,q.size/nq))
                bin = get_bin_func('q',q=q)
                ells = bin(arange(10000))
            else:
                bin = None

        return PowerSpectra(spectra=spectra,
                            cov=cov,
                            binning=bin,
                            ells=ells)
        
        
    def calibrated(self,maps,ells,weighting=1):
        """
        Return calibration factors corresponding to calibrating 
        some maps to their mean. The output of this function can then be passed to 
        PowerSpectra.lincombo
        
        Keyword arguments:
        maps -- Which maps to calibrate
        ells -- An array or a slice object corresponding to the ell range in which
                to do the calibration.
        weighting -- The weighting to use when calculating the 'miscalibration'. 
                     For example, if the spectra are C_ell's, then 
                     weighting = 2*ells+1 corresponds to map-level calibration. 
        """
        maps = list(maps)
        fid = mean([self.spectra[(a,b)][ells]*weighting for (a,b) in pairs(maps)],axis=0)
        def miscalib(calib):
            return sum(norm(self.spectra[(a,b)][ells]*weighting*calib[ia]*calib[ib] - fid) for ((a,ia),(b,ib)) in pairs(zip(maps,range(len(maps)))))
        calib = fmin(miscalib,ones(len(maps)))
        return [(newmap,[(newmap,calib[maps.index(newmap)] if newmap in maps else 1)]) for newmap in self.get_maps()]
        
    
    def lincombo(self,new_maps,normalize=True):
        """
        Return a new PowerSpectra corresponding to forming a linear combination of maps.
        
        Keyword arguments:
        new_maps -- A dictionary mapping the new maps to linear combinations of the old maps.
                    For example, {'217c': [('217',1),('353',-.14)], '143c': [('143',1),('353',-.038)]}
        normalize -- Whether to normalize the sum of the weights to 1 (i.e. keep the CMB constant)
        """
        spectra = SymmetricTensorDict([
              ((alpha2,beta2),
               sum(self.spectra[(alpha,beta)]*fa*fb for (alpha,fa) in coeff_a.items() for (beta,fb) in coeff_b.items()) 
               /(sum(fa*fb for fa in coeff_a.values() for fb in coeff_b.values()) if normalize else 1))
              for ((alpha2,coeff_a),(beta2,coeff_b)) in pairs(new_maps.items())
          ],rank=2)
        
        if self.cov:
            cov = SymmetricTensorDict([
                  (((alpha2,beta2),(gamma2,delta2)),
                   sum(
                       self.cov[((alpha,beta),(gamma,delta))]*fa*fb*fg*fd
                       for (alpha,fa) in coeff_a.items() for (beta,fb) in coeff_b.items() for (gamma,fg) in coeff_g.items() for (delta,fd) in coeff_d.items()
                   )/(sum(
                       fa*fb*fg*fd
                       for fa in coeff_a.values() for fb in coeff_b.values() for fg in coeff_g.values() for fd in coeff_d.values()
                   ) if normalize else 1))
                  for (((alpha2,coeff_a),(beta2,coeff_b)),((gamma2,coeff_g),(delta2,coeff_d))) in pairs(pairs(new_maps.items()))
              ],rank=4)
            
        else:
            cov = None
        
        return PowerSpectra(spectra,cov,self.ells,binning=self.binning)
    
    def plot(self, which=None, errorbars=True, prefix="", ax=None, fig=None, **kwargs):
        """
        Plot these powerspectra.
        
        Keyword arguments:
        which -- A list of keys to the spectra which should be plotted 
                 (default=all of them)
        errorbars -- Whether to plot error bars (default=True)
        prefix -- A prefix which shows up in the legend (default="")
        
        Other arguments:
        **kwargs -- These are passed through to the plot function.
        """
        from matplotlib.pyplot import figure
        from matplotlib.mlab import movavg
        
        if ax is None: ax = (figure(0) if fig is None else (figure(fig) if isinstance(fig,int) else fig)).add_subplot(111)
        
        if (which==None): which = pairs(self.get_maps())
        if errorbars and self.cov:
            for (a,b) in which:
                ax.errorbar(self.ells,self.spectra[(a,b)],yerr=sqrt(diagonal(self.cov[((a,b),(a,b))])),
                         label=prefix+str(a)+" x "+str(b),fmt=kwargs.pop("fmt","."),**kwargs)
        else:
            for (a,b) in which:
                ax.plot(self.ells,self.spectra[(a,b)],label=prefix+str(a)+" x "+str(b),**kwargs)


    def apply_func(self,fspec=lambda _,x: x, fcov=lambda _, x: x):
        """
        Apply an arbitrary function to each spectrum and covariance.
        
        Keyword arguments:
        fspec -- A function of (k,s) applied to the spectra where k is the 
                 powerspectrum key and s is the spectrum (default=identity)
        fspec -- A function of (k,c) applied to the covariances where k is the 
                 covariance key and c is the covariance (default=identity) 
        """
        spectra = SymmetricTensorDict([(k,fspec(k,self.spectra[k])) for k in pairs(self.get_maps()) if k in self.spectra],rank=2)
        if self.cov: cov = SymmetricTensorDict([(k,fcov(k,self.cov[k])) for k in pairs(pairs(self.get_maps())) if k in self.cov],rank=4)
        else: cov = None
        return PowerSpectra(spectra,cov,self.ells,binning=self.binning)

    def dl(self):
        """Multilies all spectra by ell(ell+1)/2/pi"""
        return self.rescaled(self.ells*(self.ells+1)/(2*pi))

    def cl(self):
        """Divides all spectra by ell(ell+1)/2/pi"""
        return self.rescaled(2*pi/self.ells/(self.ells+1))

    def rescaled(self,fac):
        """Multiplies all spectra by a constant or ell-dependent factor"""
        return self.apply_func(lambda _, spec: spec*fac, lambda _, cov: cov*outer(fac,fac))
    
    def shifted(self,delta_ell):
        """Shift the PowerSpectra over by a few ells (for plotting)."""
        return PowerSpectra(self.spectra, self.cov, self.ells+delta_ell, binning=self.binning)
    
    def unbinned(self):
        """
        Returns an unbinned (interpolated) version of this PowerSpectra.
        The covariance is lost in the transformation.
        """
        return PowerSpectra({k:interp(arange(self.ells[0],self.ells[-1]), self.ells,self[k]) for k in self.get_spectra()})
    
    def binned(self,bin):
        """ 
        Returns a binned version of this PowerSpectra. 
        
        Keyword arguments:
        bin -- Either a binning function which takes as input both vectors 
               and matrices, or a string which is passed to get_bin_func
        """
        if (type(bin)==str): bin = get_bin_func(bin)
        ps = self.apply_func(lambda _, spec: bin(spec), lambda _, cov: bin(cov))
        ps.ells = bin(self.ells)
        ps.binning = bin
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
        """
        Differences each spectra against a fiducial template
        
        Keyword arguments:
        fid -- A fiducial template which can either be the same length as the
               spectra in this PowerSpectra object, or if binning is
               provided, the a template can be at every ell and
               will be binned accordingly. 
        """
        if alen(fid)!=alen(self.ells):
            assert self.binning!=None and alen(self.binning(fid))>=alen(self.ells), "Can't figure out how diff against this powerspectrum."
            fid=self.binning(fid)[:alen(self.ells)]
        
        return self.apply_func(lambda _, cl: cl-fid, lambda _,cov: cov)
        
    def __add__(self,other):
        return PowerSpectra.sum([self,other])

    def __sub__(self,other):
        return PowerSpectra.sum([self,other.rescaled(-1)])
      
    def __mul__(self, other):
        return PowerSpectra.binary_op(self, other, (lambda x,y: x*y))

    def __div__(self, other):
        return PowerSpectra.binary_op(self, other, (lambda x,y: x/y))
      
    @staticmethod
    def binary_op(a, b, func):
        assert all(a.ells == b.ells), "Spectra need to have the same ells."
        maps = set(a.get_maps()) & set(b.get_maps())
        spectra = SymmetricTensorDict([(k,func(a[k],b[k])) for k in pairs(maps)],rank=2)
        return PowerSpectra(spectra=spectra, ells=a.ells, binning=a.binning)
        
    @staticmethod
    def sum(ps):
        """Return the sum of one or more PowerSpectra."""
        ells = [p.ells for p in ps]
        assert all(ell == ells[0] for ell in ells), "Can't add spectra with different ells."
        maps = reduce(lambda x, y: x & y, [set(p.get_maps()) for p in ps])
        spectra = SymmetricTensorDict([(k,sum(p.spectra[k] for p in ps)) for k in pairs(maps)],rank=2)
        if all([p.cov for p in ps]): cov = SymmetricTensorDict([(k,sum(p.cov[k] for p in ps)) for k in pairs(pairs(maps))],rank=4)
        else: cov=None
        return PowerSpectra(spectra,cov,ells[0],binning=ps[0].binning)


class MDict(object):
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

def read_ini(params):
    """
    Read and process an mspec ini file, returning a dictionary.
    
    Keyword arguments:
    params -- A string filename, or a dictionary for an already loaded file.
    """
    
    import imp

    try:
        mymod = imp.new_module('params1')   
        code=open(params).read()
        exec code in mymod.__dict__
        return MDict(**{k:v for k,v in mymod.__dict__.items() if not k.startswith('_')})
    except Exception as e:
        raise e


def get_bin_func(binstr,q=None):
    """
    Returns a binning function bin(x,axis=None) where x can be any rank array and 
    axis specifies the binning axis, or None for all axes.
    
    Valid binning functions are:
    WMAP -- WMAP binning at low-ell, SPT binning at high-ell
    CTP -- CTP binning
    C2 -- C2 comparison binning
    flat(x) -- Uniform bins of width x
    """
    binstr = binstr.lower()
    
    if (binstr is None or binstr=="none"): return lambda x, **kwargs: x
    
    def slice_bins(lmins,lmaxs,lslice):
        """Gets bin slice correpsonding to ell-slice 's'."""
        return slice(None if lslice.start is None else bisect_left(lmins,lslice.start),
                     None if lslice.stop is None else bisect_left(lmaxs,lslice.stop))

    def get_q_bin(q):
        ilen = lambda i: sum(1 for _ in i)
        lmins = [ilen(takewhile(lambda x: x<1e-5,qe)) for qe in q]
        lmaxs = [len(qe) - ilen(takewhile(lambda x: x<1e-5,qe[::-1])) for qe in q]
        
        def q_bin(x,axis=None):
            nq = q.shape[1]
            if type(x)==slice:
                return slice_bins(lmins=lmins,
                                  lmaxs=lmaxs,
                                  lslice=x)
            else:
                if axis==0 or axis is None and x.ndim==1: 
                    nl = min(nq,x.shape[0])
                    nqp = slice_bins(lmins, lmaxs=lmaxs, lslice=slice(0,nl))
                    return dot(q[nqp,:nl],x[:nl])
                elif axis==1: 
                    nl = min(nq,x.shape[1])
                    nqp = slice_bins(lmins, lmaxs=lmaxs, lslice=slice(0,nl))
                    return dot(x[...,:nl],q[nqp,:nl].T)
                elif axis is None:
                    nl = min(nq,x.shape[0])
                    nqp = slice_bins(lmins, lmaxs=lmaxs, lslice=slice(0,nl))
                    return dot(dot(q[nqp,:nl],x[:nl,:nl]),q[nqp,:nl].T)
                
        q_bin.q = q
        q_bin.lmaxs = lmaxs
        q_bin.lmins = lmins
        return q_bin


    def lims_to_q(lims):
        """lims is a list of (start,end), both inclusive."""
        q = zeros((len(lims),6000))
        for i,(s,e) in enumerate(lims): q[i,s:e+1] = 1. / (e-s+1)
        return q
    
    if binstr=='q':
        return get_q_bin(q)
    
    if binstr=='c2':
        return get_q_bin(np.load(os.path.join(Mrootdir,"dat/c2_binning.npy")))
    
    if binstr=="ctp":
        ctpbins=loadtxt(os.path.join(Mrootdir,"dat/CTP_bin_TT_orig"),dtype=int)
        return get_q_bin(lims_to_q(list(ctpbins[:,[1,2]])+[(l,l+200) for l in range(3001,6000,200)]))
    
    if binstr=='wmap': 
        wmapbins=array(loadtxt(os.path.join(Mrootdir,"dat/wmap_binned_tt_spectrum_7yr_v4p1.txt"))[:,:3],dtype=int)
        return get_q_bin(lims_to_q(list(wmapbins[:-2,[1,2]])+[(l,l+50) for l in range(1001,2000,50)]+[(l,l+200) for l in range(2001,4000,200)]))

    r = re.match("flat\((?:dl=)?([0-9]+)\)",binstr)
    if r!=None:
        dl=int(r.group(1))
        return get_q_bin(lims_to_q([(l,l+dl) for l in dl*arange(10000/dl)]))
    

    raise ValueError("Unknown binning function '"+binstr+"'")










def clean_signal(sig,clean,weight=1,range=slice(0,-1)):
    """
    Cleans all the power spectra in sig except for 'clean' by using 'clean' 
    as a template and minimizing the total power in the cleaned spectra. 
    """
    from scipy.optimize import fmin
    sigu = sig.unbinned()
    return {m: [(m,1),
                (clean,fmin(lambda x: sum(weight[range]*(sigu[(m,m)][range]-2*x*sigu[(m,clean)][range]+x**2*sigu[(clean,clean)][range]))/(1-2*x+x**2),.05,disp=False)[0])] 
            for m in set(sig.get_maps())-set([clean])}  


def load_beams(params,slice_lmax=False):
    """Load the beams"""

    params = read_Mspec_ini(params)
    beampow = {'bl':2,'wl':1}[params.get("beam_format","bl")] 

    if "beams_rimo" in params: 
        import pyfits
        with pyfits.open(params["beams_rimo"]) as f:
            beams = PowerSpectra({tuple(tuple([x.split('-')[0],'T']+[y.lower() for y in x.split('-')[1:]]) 
                                        for x in h.name[5:].split('X')):h.data['BEAM'][0]**beampow
                                  for h in f if h.name.startswith('BEAM') and 'BEAM' in h.columns.names})   

        return beams
    elif params["beams"].endswith(".fits"):
        return PowerSpectra.load_from_fits(params["beams"])   
    else:
        regex=re.compile(params.get("beam_regex",params.get("map_regex",def_map_regex)))
        files = [(os.path.join(params["beams"],f),[MapID(m.group(1),'T',m.group(2)) for m in regex.finditer(f)]) for f in os.listdir(params["beams"])]
        beams = SymmetricTensorDict(rank=2)
        

        
        for (f,m) in files:
            if len(m)==1: beams[(m[0],m[0])] = load_multi(f)[:int(params["lmax"]),params.get("beam_col",1)]**beampow
            elif len(m)==2: beams[(m[0],m[1])] = load_multi(f)[:int(params["lmax"]),params.get("beam_col",1)]**beampow
            
        for (m1,m2) in pairs(beams.get_index_values()):
            if (m1,m2) not in beams: 
                beams[(m1,m2)] = sqrt(beams[(m1,m1)]*beams[(m2,m2)])
    
        return PowerSpectra(beams)

    
def load_pcls(pcls,lmax=None):
    return PowerSpectra({k:loadtxt(v,usecols=[1])[slice(0,lmax)] for k,v in scan_pcls(pcls).items()})

def scan_pcls(pcls):
    return {k:v for k,v in scan_files(pcls,"(.*)__X__(.*)",transform=lambda x: tuple(tuple(xx.split('-')) for xx in x)).items() if not v.endswith("log")}

def scan_files(folder,regex,transform=None,multiple_search=False):
    if transform is None: transform=lambda x: x
    if isinstance(regex,str): regex=re.compile(regex)
    files = [(osp.join(folder,f),[transform(r.groups()) for r in regex.finditer(f)]) for f in os.listdir(folder)]
    if multiple_search:
        return {tuple(k):f for f,k in files if len(k)>0}
    else: 
        return {tuple(k[0]):f for f,k in files if len(k)>0}


class load_files(dict):
    def __init__(self,files,column=None,loadfn=None):
        super(load_files,self).__init__(files)
        self.loadfn = loadfn
        if column is not None: self.loadfn = lambda x: loadtxt(x,usecols=[1])

    def load(self):
        return {k:self.loadfn(v) for k,v in self.items()}


def equal_cross_weights(pcls):
    """
    Return a equal weighting of cross spectra of the detector maps in pcls 
    """
    fks = set((k1[0],k2[0]) for k1,k2 in pcls.spectra)
    weights = {(alpha,beta):{tuple(sorted((a,b))):1 for a,b in pcls.spectra if a!=b and a[0]==alpha and b[0]==beta} for alpha,beta in fks}
    return get_normed_weights(weights)


def get_pcl_noise(weights, pcls):
    """
    Use the given weights to construct a signal estimate from the pcls, 
    then do auto-minus-cross to give the noise estimate for each detector auto-spectrum.
    """
    pcl_sig = PowerSpectra({fpk:sum(pcls[dpk]*dpw for dpk,dpw in fpw.items()) for fpk,fpw in weights.items()})
    detmaps = set(chain(*[chain(*fpw.keys()) for fpw in weights.values()]))
    pcl_nl = {dm:(pcls[(dm,dm)] - pcl_sig[(dm[0],dm[0])]) for dm in detmaps}
    return pcl_nl


def load_kernels(kernels,lmax=None):
    glls=scan_files(kernels,'gll__(.*).npy',transform=lambda x: (lambda y: ((y[0],y[1]),(y[3],y[4])))(x[0].split('__')))
    imlls=scan_files(kernels,'imll__(.*).npy',transform=lambda x: (lambda y: (y[0],y[2]))(x[0].split('__')))
    return [SymmetricTensorDict(load_files(y,loadfn=lambda x: np.load(x)[slice(lmax),slice(lmax)]).load(),rank=r) for y,r in [(imlls,2),(glls,4)]]


def get_normed_weights(weights):
    """
    Normalize the weightings to 1.
    """
    return {fpk:{mpk:mpw/float(sum(fpw.values())) for mpk,mpw in fpw.items()} for fpk,fpw in weights.items() if len(fpw)>0}


def optimize_weights(pcls,weights=None, noise_range=(1500,2500)):
    """
    Use the given weights to call get_pcl_noise and then construct an inverse variance noise weighting. 
    """
    if weights is None: weights = equal_cross_weights(pcls)

    pcl_nl = {m:mean(nl[slice(*noise_range)]) for m,nl in get_pcl_noise(weights,pcls).items()}
   
    weights = {fpk:{mpk:1./pcl_nl[mpk[0]]/pcl_nl[mpk[0]] for mpk,mpw in fpw.items()} for fpk,fpw in weights.items()}
    return get_normed_weights(weights)

def default_mask_name_transform(m):
    return osp.basename(m).replace('.fits','')

