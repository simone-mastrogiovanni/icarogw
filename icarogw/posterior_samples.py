from .jax_pal import *
from .conversions import radec2indeces


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
        nsamps=onp.array([self.posterior_samples_dict[key].nsamples for key in self.posterior_samples_dict.keys()])
        if nparallel is None:
            nparallel=onp.min(nsamps)
        else:
            nparallel=onp.min(onp.hstack([nsamps,nparallel]))
        
        self.nparallel=nparallel
        llev=list(self.posterior_samples_dict.keys()) # Name of events
        print('Using {:d} samples from each {:d} posteriors'.format(self.nparallel,self.n_ev))
        
        self.posterior_parallel={key:onp.empty([self.n_ev,self.nparallel],
                                              dtype=self.posterior_samples_dict[llev[0]].posterior_data[key].dtype) for key in self.posterior_samples_dict[llev[0]].posterior_data.keys()}

        # Saves the posterior samples in a dictionary containing events on rows and posterior samples on columns
        for i,event in enumerate(list(self.posterior_samples_dict.keys())):
            self.posterior_samples_dict[event].cupyfy()
            len_single = self.posterior_samples_dict[event].nsamples
            rand_perm = onp.random.permutation(len_single)
            for key in self.posterior_parallel.keys():
                self.posterior_parallel[key][i,:]=jnp2onp(self.posterior_samples_dict[event].posterior_data[key])[rand_perm[:self.nparallel]]
            self.posterior_samples_dict[event].numpyfy()

        for key in self.posterior_parallel.keys():
            self.posterior_parallel[key]=onp2jnp(self.posterior_parallel[key])
            
    def update_weights(self,rate_wrapper):
        '''
        This method updates the weights associated to each injection and calculates the detected CBC rate per year in detector frame
        
        Parameters
        ----------
        
        rate_wrapper: class
            Rate wrapper from the wrapper.py module, initialized with your desired population model.
        '''
        
        self.log_weights = rate_wrapper.log_rate_PE(**{key:self.posterior_parallel[key] for key in self.posterior_parallel.keys()})
        self.sum_weights=jnp.exp(logsumexp(self.log_weights,axis=1))/self.nparallel
        self.sum_weights_squared= jnp.exp(logsumexp(2*self.log_weights,axis=1))/jnp.power(self.nparallel,2.)
        
    def get_effective_number_of_PE(self):
        '''
        Returns a vector of effective PEs for each event
        '''
        
        # Check for the number of effective sample (Eq. 2.73 document)
        
        Neff_vect=jnp.power(self.sum_weights,2.)/self.sum_weights_squared        
        Neff_vect[jnp.isnan(Neff_vect)]=0.
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


class posterior_samples(object):
    def __init__(self,posterior_dict,prior):
        '''
        Class to handle posterior samples for icarogwCAT.
        
        Parameters
        ----------
        posterior_dict: onp.array
            Dictionary of posterior samples
        prior: onp.array
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
        self.posterior_data={key:onp2jnp(self.posterior_data[key]) for key in self.posterior_data}
        
    def numpyfy(self):
        ''' Converts all the posterior samples to numpy'''
        self.posterior_data={key:jnp2onp(self.posterior_data[key]) for key in self.posterior_data}
        
    def add_counterpart(self,z_EM,ra,dec):
        '''
        This method adds an EM counterpart to the posterior samlples. It practically
        selects all the posterior samples falling in the pixel of the EM coutnerpart. Note that you should have
        already pixelized the posterior by running the pixelize method.
        
        Parameters
        ----------
        z_EM: jnp.array
            Samples of credible cosmological redshifts inferred from EM counterparts.
            This should already include all the uncertainties, e.g. peculiar motion
        ra: float
            Right ascension of the EM counterpart in radians.
        dec: float
            declination of the EM counterpart in radians.
        '''
        
        idx = radec2indeces(ra,dec,self.nside)
        select = jnp.where(self.posterior_data['sky_indices']==idx)[0]
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
        prob = jnp.exp(logw)
        prob/=prob.sum()
        idx = jnp.random.choice(len(self.posterior_data['prior']),replace=replace,p=prob)
        return {key:self.posterior_data[key][idx] for key in list(self.posterior_data.keys())}
        
        
        
        
        
