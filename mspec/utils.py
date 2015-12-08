import re, os
from itertools import combinations_with_replacement, product
from numpy import *
import itertools
import traceback
from collections import namedtuple
from numpy.ma.core import sort
from multiprocessing.dummy import Pool
import threading

""" When loading files via load_multi, this is the default type to convert them to """
def_dtype = float64
    
""" The Mspec root directory (used for loading some Mspec-wide files) """
Mrootdir = os.path.abspath(os.path.dirname(__file__))

""" If True, don't even try to load the mpi4py library or use MPI at all """
NOMPI = False

""" Don't print log messages."""
log_quiet = False

def mspec_log(msg,rootlog=False,mpisafe=False):
    if not log_quiet and (not rootlog or is_mpi_master()):
        sig = ['?' if mpisafe else ('%%.%ii'%(int(floor(log10(get_mpi_size()))+1)))%get_mpi_rank(), str(os.getpid())]
        if not (threading.current_thread().__class__.__name__ == '_MainThread'): 
            sig.append(threading.currentThread().getName())
        print '\033[95m%s:\033[0m %s'%('-'.join(sig),msg)


def test_sym(m):
    """ Test the symmetry of a matrix"""
    return max(x for x in ((abs(m-m.T))/((m+m.T)/2.)).flatten() if isfinite(x))


def get_num_threads():
    """
    Returns the number of threads to use by reading the environment 
    variable OMP_NUM_THREADS, or if that's not present the 
    number of cores on the current processor.
    """
    try:
        from multiprocessing import cpu_count
        if os.environ.has_key("OMP_NUM_THREADS"): return int(os.environ["OMP_NUM_THREADS"])
        else: return cpu_count()
    except:
        return 1



def get_mpi():
    """ Return (rank,size,comm) """
    if NOMPI or len([l[2] for l in traceback.extract_stack() if l[2] == 'mpi_map']) > 1:
        (rank,size,comm) = (0,1,None)
    else:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            (rank,size) = (comm.Get_rank(),comm.Get_size())
        except Exception as e:
            (rank,size,comm) = (0,1,None)
            
    return namedtuple('mpi',['rank','size','comm'])(rank,size,comm)

def is_mpi_master(): return get_mpi_rank()==0
def get_mpi_rank(): return get_mpi().rank
def get_mpi_size(): return get_mpi().size




def mpi_consistent(value):
    """Returns the value that the root process provided."""
    comm = get_mpi().comm
    if comm==None: return value
    else: return comm.bcast(value)

def mpi_map(function,sequence,pool=None,distribute=True):
    """
    A map function parallelized with MPI. If this program was called with mpiexec -n $NUM, 
    then partitions the sequence into $NUM blocks and each MPI process does the rank-th one.
    Note: If this function is called recursively, only the first call will be parallelized

    Keyword arguments:
    distribute -- If true, every process receives the answer
                  otherwise only the root process does (default=True)
    """
    (rank,size,comm) = get_mpi()

    sequence = mpi_consistent(sequence)

    if pool is not None: _map=pool.map
    else: _map=map

    if (size==1):
        return _map(function,sequence)
    else:
        if (distribute):
            return flatten(comm.allgather(_map(function, partition(sequence,size)[rank])))
        else:
            if (rank==0):
                return flatten(comm.gather(_map(function, partition(sequence,size)[rank])))
            else:
                comm.gather(_map(function, partition(sequence,size)[rank]))
                return []

def mpi_map2(function,sequence,pool=None,distribute=True,nthreads=None):
    """
    Map using mpi4py_map
    """
    import mpi4py_map
    (rank,size,comm) = get_mpi()

    sequence = mpi_consistent(sequence)
    if pool is None:
        return mpi4py_map.map(function,sequence)
    else:
        def thread_map(sequence):
            return pool.map(function,sequence)    
        return flatten(mpi4py_map.map(thread_map,chunks(sequence,nthreads or get_num_threads())))



def mpi_map_array(function,sequence):
    (rank,size,comm) = get_mpi()
    sequence = mpi_consistent(sequence)
    nper = int(ceil(float(len(sequence))/size))
    buf = zeros([nper]+list(shape))
    for i,x in enumerate(sequence[rank*nper:(rank+1)*nper]): buf[i]=function(x)
    allbuf = zeros([size,nper]+list(shape))
    comm.Allgather(buf,allbuf)
    return allbuf.reshape([size*nper]+list(shape))[:len(sequence)]


def load_multi(path):
    """ Load either path or path.npy """
    if os.path.exists(path+".npy"): return array(load(path+".npy"),dtype=def_dtype)
    elif os.path.exists(path): return loadtxt(path,dtype=def_dtype)
    else: raise IOError("No such file or directory: "+path+".npy or "+path)

def save_multi(path,dat,npy=True, mkdir=True):
    """ Save either path or path.npy """
    if mkdir and not os.path.exists(os.path.dirname(path)): 
        print "Creating folder '"+os.path.dirname(path)+"'"
        os.makedirs(os.path.dirname(path))
    if npy: save(path, array(dat,dtype=float32))
    else: savetxt(path,dat)
    
def pairs(*arg):
    """
    pairs(list)
        Returns a list of tuples corresponding to all (order unimportant) pairs of elements from list
        
    pairs(list1,list2)
        Returns a list of tuples corresponding to all (order unimportant) pairs of one element from each list
    """
    if (len(arg)==1): return list(combinations_with_replacement(arg[0],2))
    elif (len(arg)==2): return list(set(product(arg[0],arg[1])))
    else: raise ValueError("pairs expected one or two lists as an argument")

def flatten(l):
    """Returns a list of lists joined into one"""
    return list(itertools.chain(*l))

def partition(list, n):
    """Partition list into n nearly equal sublists"""
    division = len(list) / float(n)
    return [list[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

def chunks(l, n):
    """Partition list into length-n sublists"""
    return [l[i:i+n] for i in range(0, len(l), n)]

def str2bool(s):
    """Convert a string to boolean"""
    if type(s)==bool: return s
    elif s.lower() in ["t","true"]: return True
    elif s.lower() in ["f","false"]: return False
    else: raise ValueError("Couldn't convert '"+str(s)+"' to boolean.")


def cust_legend(opts,labels,**kwargs):
    """A custom legend"""
    from matplotlib.pyplot import legend, Line2D
    legend([Line2D([0],[0],**o) for o in opts],labels,**kwargs)

def corrify(m):
    """Convert a covariance matrix into a correlation matrix"""
    m2=m.copy()
    for i in range(alen(m)): 
        m2[i,:]/=sqrt(m[i,i])
        m2[:,i]/=sqrt(m[i,i])
    return m2


def try_type(v):
    """If v is a numerical string, returns float(v), or recursively try so on a list"""
    if (type(v)==list): return map(try_type,v)
    if (type(v)!=str): return v
    try:
        return float(v)
    except ValueError:
        if (v.lower() in ["t","true"]): return True
        elif (v.lower() in ["f","false"]): return False
        return v

def confint2d(hist,which):
    """Return """
    H=sort(hist.ravel())[::-1]
    sumH=sum(H)
    cdf=array([sum(H[H>x])/sumH for x in H])
    return interp(which,cdf,H)


"""
cookb_signalsmooth.py

from: http://scipy.org/Cookbook/SignalSmooth
"""

import numpy as np

def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]

