import sys, os, re
import numpy as np
from numpy import array, shape, loadtxt, log, genfromtxt, cov, sqrt, diag, vstack, sum, average, hstack, mean
from random import random
from itertools import product
from matplotlib.pyplot import plot, hist


"""

    Runs mcmc chains based off of parameter files and user defined likelihood functions.
    
"""





def mcmc(start,lnl,init_fn=[],derived_fn=[],step_fn=[]):
    """
    
    Run an MCMC chain. 
    
    Parameters:
        start - dictionary of parameter key/values or a filename
        lnl - a function (or list of functions) of (dict:params) which returns the negative log-likelihood
        init_fn = a function (or list of functions) of (dict:params) which is called before starting the chain
        derived_fn = a function (or list of functions) of (dict:params) which calculates and adds derived parameters 
        step_fn - a function (or list of functions) of (dict:params, dict:samples) which is called after every step
    
    Returns:
        A dictionary of samples. There is a key for every parameter which was varied, 
        one for every derived parameter, and one for "lnl" and "weight"
        
    Example:
        samples = mcmc({"x":"0 [-10 10 1]",'samples':10000}, lambda p: p["x"]**2/2)
        
        plot(samples["x"])
    """
    
    
    #Make lists of the input functions if they aren't
    [lnl,init_fn,derived_fn,step_fn] = map(lambda x: x if type(x)==list else [x],[lnl,init_fn,derived_fn,step_fn])

    #Read and process the starting parameters
    if (type(start)==str): start=read_params(start)
    cur_params = get_processed_params(start)
    
    #Call initialization functions
    for fn in derived_fn: fn(cur_params)
    for fn in init_fn: fn(cur_params)
    
    #Initialize starting sample
    samples = {"lnl":[sum([l(cur_params) for l in lnl])],"weight":[1]}
    for name in get_outputted(cur_params): samples[name]=[cur_params[name]]
    
    #Initialize file if were writing to a file
    if (cur_params.get("file_root","")!=""):
        output_file = open(cur_params["file_root"],"w")
        output_file.write("# lnl weight "+" ".join(get_outputted(cur_params))+"\n")
        output_to_file = True
    else:
        output_to_file = False
    
    #Start the MCMC
    for sample_num in range(int(start.get("samples",100))):
        test_params = propose_step_gaussian(cur_params)
        
        for fn in derived_fn: fn(test_params)
        
        # Check min/max bounds
        test_lnl = 0
        for name in get_varied(test_params): 
            if (not (test_params["*"+name][1] < test_params[name] < test_params["*"+name][2])): test_lnl = np.inf
        
        #Get likelihood
        if (test_lnl != np.inf): test_lnl = sum([l(test_params) for l in lnl])
                
#        mcmc_log(cur_params,"Like="+str(test_lnl)+" Ratio="+str(np.mean(1./array(samples["weight"])))+" Sample="+str(dict([(name,test_params[name]) for name in get_outputted(cur_params)]))) 

        if (log(random()) < samples["lnl"][-1]-test_lnl):

            #Add to file (which lags samples by one accepted sample so we get the weight right)
            if (output_to_file): 
                output_file.write(" ".join([str(samples[name][-1]) for name in ["lnl","weight"]+get_outputted(cur_params)])+"\n") 
                output_file.flush()
            
            
            #Add to samples
            for name in get_outputted(test_params): samples[name].append(test_params[name])
            samples["lnl"].append(test_lnl)
            samples["weight"].append(1)
            
            cur_params = test_params

        else:
            samples["weight"][-1] += 1
            
            
        #Call step function provided by user
        for fn in step_fn: fn(cur_params,samples)
            
            
    if (output_to_file): output_file.close()
    
    for k in samples.keys(): samples[k]=array(samples[k])

    return samples



def mpi_mcmc(start,lnl,init_fn=[],derived_fn=[],step_fn=[]):  
    """
    
    Runs an MCMC chain on an MPI cluster using (processes-1) workers and 1 master.
    
    Example:
    
        mpiexample.py:
            import mcmc
            mpi_mcmc({"*x":[0,-10,10,1],'samples':10000,"file_root":"chain"}, lambda p: p["x"]**2/2)
    
        mpiexec -n 8 python mpiexample.py
    
    
    Internally, every delta_send_samples number of steps, the worker processes communicate new steps to the master process. 
    They then immediately ask the master process for a dictionary of parameters with which to update the local copy (for example, a new proposal covariance).
    
    The master process aggregates the samples from all of the chains, calculating a proposal covariance and sending it out to the
    worker processes whenever necessary. 
    
    """

    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()-1

    if (size==0): 
        mcmc(start,lnl,init_fn=init_fn,derived_fn=derived_fn,step_fn=step_fn)
        return 

    if (type(start)==str): start=read_params(start)
    start = get_processed_params(start)
    start["samples"]/=size
    if "file_root" in start: start["file_root"]+=("_"+str(rank))

    #Make lists of the input functions if they aren't
    [lnl,init_fn,derived_fn,step_fn] = map(lambda x: x if type(x)==list else [x],[lnl,init_fn,derived_fn,step_fn])

    delta_send_samples = 50

    def wslice(samples,delta):
        """
        Gets the last delta non-unique samples, possibly slicing one weighted sample
        """
        weights = samples["weight"]
        nws = len(filter(lambda x: x,np.cumsum(weights[-delta:])-sum(weights[-delta:])>-delta))
        sl = dict([(name,samples[name][-nws:]) for name in samples.keys()])
        sl["weight"][0]-=(sum(weights[-nws:])-delta)
        return sl
        
    def mpi_step_fn(params,samples):
        """ 
        This is the worker process code.
        """
        if (sum(samples["weight"])%delta_send_samples==0):
            mcmc_log(params,"Chain "+str(rank)+": steps="+str(sum(samples["weight"]))+" approval="+str(np.mean(1./array(samples["weight"])))+" best:"+str(min(samples["lnl"][1:])))
            comm.send((rank,wslice(samples,delta_send_samples)))    
            new_params = comm.recv()
            if (new_params!=None):  
                mcmc_log(params,"Chain "+str(rank)+": New proposal "+str(zip(get_varied(params),sqrt(diag(new_params["$COV"])))))
                params.update(new_params)

    def mpi_master_fn(samples,source):
        """ 
        
        This is the master process code.
        
        samples - a list of samples aggregated from all of the chains
        source - the rank of the chain which is requesting updated parameters
        
        """
        
        if (start["$UPDATECOV"] and (lambda x: x>=2000 and x%delta_send_samples==0)(sum(samples[source-1]["weight"])) and start["$RMIN"]<gelman_rubin_R(samples)<start["$RMAX"]): 
            return {"$COV":get_new_proposal(samples, start)}
        else: 
            return None

    if (rank==0):
        samples = [dict([(name,[]) for name in get_varied(start)+["lnl","weight"]]) for i in range(size)]
        finished = [False]*size
        while (not all(finished)):
            (source,new_samples)=comm.recv(source=MPI.ANY_SOURCE)
            if (new_samples!=None): 
                for name in get_varied(start)+['lnl','weight']: 
                    samples[source-1][name]+=(lambda x: x if type(x)==list else [x])(new_samples[name])
            else:
                finished[source-1]=True
                
            comm.send(mpi_master_fn(samples,source),dest=source)
        
    else:
        
        mcmc(start,lnl,init_fn=init_fn,step_fn=step_fn+[mpi_step_fn],derived_fn=derived_fn)["weight"]
        comm.send((rank,None))


def mcmc_log(params,message):
    if (params["$MCMC_VERBOSE"]): print message


def get_processed_params(params):
    """
    Process the parameters string key-value pairs given in params.
    
    -Make values into lists (if more than one token)
    -Attempt to convert everything to float/boolean
    -For all varied parameters (ones which have [MIN MAX WIDTH]), remove the [], add a *name, and add them to $VARIED
    -Either load the covariance from a file, or generate it from the WIDTHs
    """
    
    if (params.get("$PROCESSED",False)): return params
    
    varied_params = []
    processed = {}
    
    for (k,v) in params.items():
        if (type(v)==str):
            r = re.search("(.*?)\[(.*?)\]$",v)
            if (r==None):
                pv=try_type(v.split())
            else:
                pv=float(r.groups()[0])
                processed["*"+k]=map(float,[r.groups()[0]]+r.groups()[1].split())
                varied_params.append(k)
            if (type(pv)==list and len(pv)==1): pv=pv[0]
        else:
            pv = v
            
        processed[k] = pv
        
    processed["$VARIED"] = varied_params 
    processed["$OUTPUT"] = varied_params + params.get("derived","").split()
    initialize_covariance(processed)
    processed["$COV_FAC"]=float(processed.get("proposal_scale",2.4))
    processed["$MCMC_VERBOSE"]=processed.get("mcmc_verbose",False)
    processed["$UPDATECOV"]=processed.get("proposal_update",True)
    [processed["$RMIN"],processed["$RMAX"]] = processed.get("proposal_update_minmax_R",[0,np.inf])
    processed["$PROCESSED"] = True
    
    return processed

def try_type(v):
    """
    If v is a numerical string, returns float(v) 
    """
    if (type(v)==list): return map(try_type,v)
    if (type(v)!=str): return v
    try:
        return float(v)
    except ValueError:
        if (v.lower() in ["t","true"]): return True
        elif (v.lower() in ["f","false"]): return False
        return v
            

def get_varied(params):
    return params["$VARIED"]

def get_outputted(params):
    return params["$OUTPUT"]

def propose_step_gaussian(params):
    """
    Take a gaussian step in $VARIED according to $COV
    """
    varied_params = get_varied(params)
    cov = params["$COV"]
    nparams = len(varied_params)
    if (shape(cov)!=(nparams,nparams)):
        raise ValueError("Covariance not the same length as number varied parameters.")
    dxs = np.random.multivariate_normal([0]*nparams,cov) * params["$COV_FAC"]
    propose = params.copy()
    for (name,dx) in zip(varied_params,dxs): propose[name]+=dx
    return propose


def initialize_covariance(params):
    """
    Load the covariance, defaulting to diagonal entries from the WIDTH of each parameter
    """
    
    v=params.get("proposal_matrix","")
    if (v==""): prop_names, prop = [], None
    else: 
        with open(v) as file:
            prop_names = re.sub("#","",file.readline()).split()
            prop = genfromtxt(file)

    params["$COV"] = np.diag([params["*"+name][3]**2 for name in get_varied(params)])
    common = set(get_varied(params)) & set(prop_names)
    if common: 
        idxs = zip(*(list(product([ps.index(n) for n in common],repeat=2)) for ps in [get_varied(params),prop_names]))
        for ((i,j),(k,l)) in idxs: params["$COV"][i,j] = prop[k,l]
        
   
   
def get_covariance(data,weights=None):
    if (weights==None): return np.cov(data.T)
    else:
        mean = np.sum(data.T*weights,axis=1)/np.sum(weights)
        zdata = data-mean
        return np.dot(zdata.T*weights,zdata)/(np.sum(weights)-1)


def get_new_proposal(samples,params):
    nchains = len(samples)
    nsamp = [len(s['weight']) for s in samples]
    data = array([reduce(lambda a,b: a+b,[samples[i][name][:nsamp[i]/2] for i in range(nchains)]) for name in get_varied(params)]).T
    weights = array(reduce(lambda a,b: a+b,[samples[i]["weight"][:nsamp[i]/2] for i in range(nchains)]))
    return get_covariance(data, weights)
    
def gelman_rubin_R(samples):
    nchains = len(samples)
    nsamp = [len(s['weight']) for s in samples]
    return 1
    

class Chain(dict):
    def params(self): return set(self.keys())-set(["lnl","weight"])
    def sample(self,s,keys=None): return Chain((k,self[k][s]) for k in (keys if keys else self.keys()))
    def matrix(self,params=None): return vstack([self[p] for p in (params if params else self.params())]).T
    def cov(self,params=None): return get_covariance(self.matrix(params), self["weight"])
    def mean(self,params=None): return average(self.matrix(params),axis=0,weights=self["weight"])
    def std(self,params=None): return sqrt(average((self.matrix(params)-self.mean(params))**2,axis=0,weights=self["weight"]))
    def like1d(self,p,bins=30): hist(self[p],weights=self["weight"],bins=bins,normed=True)
    def like2d(self,p1,p2): raise NotImplementedError()
    def acceptance(self): return mean(1./self["weight"])
    def savecov(self,file,params=None):
        if not params: params = self.params()
        with open(file) as f:
            f.write("# "+" ".join(params))
            savetxt(f,self.cov(params))
        
class Chains(list):
    def burnin(self,nsamp): return Chains(c.sample(slice(nsamp,-1)) for c in self)
    def join(self): return Chain((k,hstack([c[k] for c in self])) for k in self[0].keys())
    def plot(self,param): 
        for c in self: plot(c[param])
    
def load_chain(filename):
    def load_one_chain(filename):
        with open(filename) as file:
            names = re.sub("#","",file.readline()).split()
            data = genfromtxt(file)
            
        return Chain([(name,data[:,i]) for (i,name) in enumerate(names)])
        
    dir = os.path.dirname(filename)
    files = [os.path.join(dir,f) for f in os.listdir(dir) if f.startswith(os.path.basename(filename))]
    if len(files)==1: return load_one_chain(files[0])
    elif len(files)>1: return Chains(load_one_chain(f) for f in files)
    else: raise IOError("File not found: "+filename) 


