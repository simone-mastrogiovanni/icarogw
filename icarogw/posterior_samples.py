from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn
from .conversions import radec2indeces

# LVK Reviewed
class posterior_samples_catalog(object):
    
    def __init__(self,posterior_samples_dict):
        '''
        A class to handle a list of posterior samples objects
        
        Parameters
        ----------
        posterior_samples_dict: dictionary
            Dictionary of posterior_samples classes
        '''

        self.posterior_samples_dict = posterior_samples_dict
        self.n_ev = len(posterior_samples_dict)
    
    def build_parallel_posterior(self,nparallel=None):
        '''
        Build a matrix of GW parameters by selecting random samples from each posterior
        
        Parameters
        ----------
        nparallal: int
            Number of posterior samples to select, if None it will select the maximum common number
        '''
        
        
        # Saves the minimum number of samples to use per event
        nsamps=np.array([self.posterior_samples_dict[key].nsamples for key in self.posterior_samples_dict.keys()])
        if nparallel is None:
            nparallel=np.min(nsamps)
        else:
            nparallel=np.min(np.hstack([nsamps,nparallel]))
        
        self.nparallel=nparallel
        llev=list(self.posterior_samples_dict.keys()) # Name of events
        print('Using {:d} samples from each {:d} posteriors'.format(self.nparallel,self.n_ev))
        
        key = list(self.posterior_samples_dict[llev[0]].posterior_data.keys())[0]
        xp = get_module_array(self.posterior_samples_dict[llev[0]].posterior_data[key])
        
        self.posterior_parallel={key:xp.empty([self.n_ev,self.nparallel],
                                              dtype=self.posterior_samples_dict[llev[0]].posterior_data[key].dtype) for key in self.posterior_samples_dict[llev[0]].posterior_data.keys()}

        # Saves the posterior samples in a dictionary containing events on rows and posterior samples on columns
        for i,event in enumerate(list(self.posterior_samples_dict.keys())):
            len_single = self.posterior_samples_dict[event].nsamples
            rand_perm = xp.random.permutation(len_single)
            for key in self.posterior_parallel.keys():
                self.posterior_parallel[key][i,:]=self.posterior_samples_dict[event].posterior_data[key][rand_perm[:self.nparallel]]
            self.posterior_samples_dict[event].numpyfy() # Big data is forced to be on CPU
            
    def cupyfy(self):
        ''' Converts all the posterior samples to cupy'''
        self.posterior_parallel={key:np2cp(self.posterior_parallel[key]) for key in self.posterior_parallel}
        
    def numpyfy(self):
        ''' Converts all the posterior samples to numpy'''
        self.posterior_parallel={key:cp2np(self.posterior_parallel[key]) for key in self.posterior_parallel}
        
            
    def update_weights(self,rate_wrapper):
        '''
        This method updates the weights associated to each injection and calculates the detected CBC rate per year in detector frame
        
        Parameters
        ----------
        
        rate_wrapper: class
            Rate wrapper from the wrapper.py module, initialized with your desired population model.
        '''

        self.log_weights = rate_wrapper.log_rate_PE(self.posterior_parallel['prior'],
                                                    **{key:self.posterior_parallel[key] for key in rate_wrapper.PEs_parameters})
        xp = get_module_array(self.log_weights)
        sx = get_module_array_scipy(self.log_weights)
        kk = list(self.posterior_parallel.keys())[0]
        self.sum_weights=xp.exp(sx.special.logsumexp(self.log_weights,axis=1))/self.nparallel
        self.sum_weights_squared= xp.exp(sx.special.logsumexp(2*self.log_weights,axis=1))/xp.power(self.nparallel,2.)
        
    def get_effective_number_of_PE(self):
        '''
        Returns a vector of effective PEs for each event
        '''
        
        # Check for the number of effective sample (Eq. 2.73 document)
        xp = get_module_array(self.sum_weights)
        Neff_vect=xp.power(self.sum_weights,2.)/self.sum_weights_squared        
        Neff_vect[xp.isnan(Neff_vect)]=0.
        return Neff_vect
    
    def pixelize(self,nside):
        '''
        Pixelize all the posteriors present in the catalog
        
        Parameters
        ----------
        nside: int
            Nside to pass to healpy
        '''
        for i,event in enumerate(list(self.posterior_samples_dict.keys())):
            self.posterior_samples_dict[event].pixelize(nside)
            
    def reweight_PE(self,rate_wrapper,Nsamp,replace=True):
        '''
        Reweights the posterior samples with a given rate model
        
        Parameters
        ----------
        rate_model: class
            Rate model from the wrappers module
        Nsamp: int
            Number of posterior samples to draw
        replace: bool
            Replace the injections with a copy once drawn
            
        Returns
        -------
        Dictionary of dictionaries containig PE
        '''
        name_ev = list(self.posterior_samples_dict.keys())
        return {key:self.posterior_dict[key].reweight_PE(rate_wrapper,Nsamp,replace=replace) for key in name_ev}

# LVK Reviewed
class posterior_samples(object):
    def __init__(self,posterior_dict,prior):
        '''
        Class to handle posterior samples for icarogwCAT.
        
        Parameters
        ----------
        posterior_dict: np.array
            Dictionary of posterior samples
        prior: np.array
            Prior to use in order to reweight posterior samples written in the same variables that you provide, e.g. if you provide d_l and m1d, then p(d_l,m1d)
        '''
        self.posterior_data={key: posterior_dict[key] for key in posterior_dict.keys()}
        self.posterior_data['prior']=prior
        self.nsamples=len(prior)
        
    def pixelize(self,nside):
        '''
        Pixelize the Ra and DEC samples
        
        Parameters
        ----------
        nside: integer
            Nside for healpy
        '''
        self.posterior_data['sky_indices'] = radec2indeces(self.posterior_data['right_ascension'],self.posterior_data['declination'],nside)
        self.nside=nside
        
    def cupyfy(self):
        ''' Converts all the posterior samples to cupy'''
        self.posterior_data={key:np2cp(self.posterior_data[key]) for key in self.posterior_data}
        
    def numpyfy(self):
        ''' Converts all the posterior samples to numpy'''
        self.posterior_data={key:cp2np(self.posterior_data[key]) for key in self.posterior_data}
        
    def add_counterpart(self,z_EM,ra,dec):
        '''
        This method adds an EM counterpart to the posterior samlples. It practically
        selects all the posterior samples falling in the pixel of the EM coutnerpart. Note that you should have
        already pixelized the posterior by running the pixelize method.
        
        Parameters
        ----------
        z_EM: xp.array
            Samples of credible cosmological redshifts inferred from EM counterparts.
            This should already include all the uncertainties, e.g. peculiar motion
        ra: float
            Right ascension of the EM counterpart in radians.
        dec: float
            declination of the EM counterpart in radians.
        '''
        xp = get_module_array(ra)
        idx = radec2indeces(ra,dec,self.nside)
        select = xp.where(self.posterior_data['sky_indices']==idx)[0]
        print('There are {:d} samples in the EM counterpart direction'.format(len(select)))
        self.posterior_data={key: self.posterior_data[key] for key in self.posterior_data.keys()}
        self.posterior_data['z_EM'] = z_EM
        self.nsamples=len(self.posterior_data['sky_indices'])
        
    def reweight_PE(self,rate_wrapper,Nsamp,replace=True):
        '''
        Reweights the posterior samples with a given rate model
        
        Parameters
        ----------
        rate_model: class
            Rate model from the wrappers module
        Nsamp: int
            Number of posterior samples to draw
        replace: bool
            Replace the injections with a copy once drawn
            
        Returns
        -------
        Dictionary containing the reweighted PE
        '''
        
        logw = rate_wrapper.log_rate_PE(**{key:self.posterior_data[key] for key in self.posterior_data.keys()})
        xp = get_module_array(logw)
        prob = xp.exp(logw)
        prob/=prob.sum()
        idx = xp.random.choice(len(self.posterior_data['prior']),replace=replace,p=prob)
        return {key:self.posterior_data[key][idx] for key in list(self.posterior_data.keys())}
