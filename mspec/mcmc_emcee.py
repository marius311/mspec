import emcee, random, re
from utils import *

class NamedEnsembleSampler(emcee.EnsembleSampler):
    def __init__(self, nwalkers, params, lnprob, extra_params={},**kwargs):
        self.params = params
        dim = len(params)
        lnprob2 = lambda x,*args: -lnprob(dict(extra_params,**dict(zip(params,x))),*args)
        super(NamedEnsembleSampler, self).__init__(nwalkers,dim,lnprob2,**kwargs)
        
    def sample(self, pos0, **kwargs):
        return super(NamedEnsembleSampler, self).sample(vstack([pos0[k] for k in self.params]).T,**kwargs)


def get_varied(params): return params["$VARIED"]
def get_outputted(params): return params["$OUTPUT"]

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

            
def get_mcmc_params(params):
    """
    Process the parameters string key-value pairs given in params.
    
    -Make values into lists (if more than one token)
    -Attempt to convert everything to float/boolean
    -For all varied parameters (ones which have [MIN MAX WIDTH]), remove the [], add a *name, and add them to $VARIED
    -Either load the covariance from a file, or generate it from the WIDTHs
    """
    
    if type(params)==str: params = read_ini(params)
    if (params.get("$PROCESSED",False)): return params
    
    varied_params = []
    processed = {}
    
    for (k,v) in params.items():
        pv = v
        if (type(v)==str):
            r = re.search("({0})\s\[\s?({0})\s({0})\s({0})\s?\]".format("[0-9.eE+-]+?"),v)
            if (r==None):
                try: 
                    pv = literal_eval(v)
                    if isinstance(pv, list): pv=array(pv)
                except: pv = try_type(v)
            else:
                pv=float(r.groups()[0])
                processed["*"+k]=map(float,r.groups())
                varied_params.append(k)
                
        processed[k] = pv
        
    processed["$VARIED"] = varied_params 
    processed["$OUTPUT"] = varied_params + params.get("derived","").split()
    initialize_covariance(processed)
    processed["$MCMC_VERBOSE"]=processed.get("mcmc_verbose",False)
    processed["$PROCESSED"] = True
    
    return processed


def mcmc(start_params, lnl, init=None, mpi=False):
    p = get_mcmc_params(start_params)
    if init!=None: init(p)
    nwalkers = p.get('walkers',100)
    nsamp = p.get('samples',10000)
    
    lnl2 = lambda p: lnl(p) if all([p['*'+k][1]<p[k]<p['*'+k][2] for k in get_varied(p)]) else inf
    
    sampler=NamedEnsembleSampler(nwalkers,get_varied(p),lnl2,extra_params=p,pool=namedtuple('pool',['map'])(mpi_map) if mpi else None)
    
    p0=mpi_consistent(dict(zip(get_varied(p),np.random.multivariate_normal([p[k] for k in get_varied(p)],p['$COV'],size=nwalkers).T)))

    if 'chain' in p and is_mpi_master(): file=open(p['chain'],'w')
    else: file=None
    
    if file: file.write('#'+' '.join(get_varied(p))+'\n')
    for i,(pos,lnprob,state) in enumerate(sampler.sample(p0,iterations=nsamp/nwalkers),1):
        if file:
            savetxt(file,pos)
            file.flush()
            
        if is_mpi_master(): print 'steps=%i approval=%.3f best=%.3f'%(i*nwalkers,sampler.acceptance_fraction.mean(),-sampler.lnprobability[:,:i].max())
    
    if file: file.close()
    
    return sampler