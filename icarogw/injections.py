from .jax_pal import *
import copy as cp
from .conversions import  radec2indeces

class injections(object):
    def __init__(self,injections_dict,prior,ntotal,Tobs):

        '''
        This class is used to manage a list of detected injections to calculate
        GW selection effects. This class uses injections which are given in source frame.
        injections_dict: jnp.array
            Dictionary containing the variables with which you want to evaluate the injections.
        prior: jnp.array
            Used prior draws for injections, same as the ones in injections dict p(mass1,mass2,dl,ra,dec)
        ntotal: float
            Total number of simulated injections (detected and not). This is necessary to compute the expected number of detections
        Tobs: float
            Length of time for the run in years (used to calculate rates)
        '''
        
        # Saves what you provided in the class
        self.injections_data_original={key:injections_dict[key] for key in injections_dict.keys()}
        self.injections_data={key:injections_dict[key] for key in injections_dict.keys()}
        self.prior_original=cp.deepcopy(prior)
        self.prior=cp.deepcopy(prior)
        self.detection_index=jnp.ones_like(prior,dtype=bool)
        self.ntotal=ntotal
        self.Tobs=Tobs
        
    def update_cut(self,detection_index):
        ''' Update the selection cut and the injections that you are able to detect
        
        Parameters
        ----------
        detection_index: jnp.array
            Array with True where you detect the injection and False otherwise
        '''
        self.detection_index=detection_index
        self.injections_data={key:self.injections_data_original[key][detection_index] for key in self.injections_data_original.keys()}
        self.prior=self.prior_original[detection_index]
        
    def cupyfy(self):
        ''' Converts all the posterior samples to cupy'''
        self.injections_data_original={key:onp2jnp(self.injections_data_original[key]) for key in self.injections_data_original.keys()}
        self.injections_data={key:onp2jnp(self.injections_data[key]) for key in self.injections_data_original.keys()}
        self.prior_original=onp2jnp(self.prior_original)
        self.prior=onp2jnp(self.prior)
        
    def numpyfy(self):
        ''' Converts all the posterior samples to numpy'''
        self.injections_data_original={key:jnp2onp(self.injections_data_original[key]) for key in self.injections_data_original.keys()}
        self.injections_data={key:jnp2onp(self.injections_data[key]) for key in self.injections_data_original.keys()}
        self.prior_original=jnp2onp(self.prior_original)
        self.prior=jnp2onp(self.prior)
        
    def effective_injections_number(self):
        ''' Get the effetive number of injections
        '''
        mean = jnp.exp(logsumexp(self.log_weights))/self.ntotal
        var = jnp.exp(logsumexp(2*self.log_weights))/(self.ntotal**2)-(mean**2)/self.ntotal
        return (mean**2)/var
    
    def pixelize(self,nside):
        self.nside=nside
        self.injections_data_original['sky_indices'] = radec2indeces(
            jnp2onp(self.injections_data_original['right_ascension']),
            jnp2onp(self.injections_data_original['declination']),nside)
        self.injections_data['sky_indices'] = radec2indeces(
            jnp2onp(self.injections_data['right_ascension']),
            jnp2onp(self.injections_data['declination']),nside)
        
        if isinstance(self.prior,onp.ndarray):
            self.numpyfy()
            
    def update_weights(self,rate_wrapper):
        '''
        This method updates the weights associated to each injection and calculates the detected CBC rate per year in detector frame
        
        Parameters
        ----------
        
        rate_wrapper: class
            Rate wrapper from the wrapper.py module, initialized with your desired population model.
        '''
        
        self.log_weights = rate_wrapper.log_rate_injections(self.prior,**{key:self.injections_data[key] for key in self.injections_data.keys()})
        self.pseudo_rate = jnp.exp(logsumexp(self.log_weights))/self.ntotal # Eq. 1.5 on the overleaf documentation
        
    def expected_number_detections(self):
        '''
        This method calculates the expected number of CBC detectable in a given time. It uses the Tobs initialized for the injections class
        '''
        
        return self.Tobs*self.pseudo_rate
    
    def return_reweighted_injections(self,Nsamp,replace=True):
        '''
        Return a set of injections detected by reweighting with the loaded rate model
        
        Parameters
        ----------
        
        Nsamp: int
            Samples to generate
        replace: bool
            Replace the injections with a copy once drawn
            
        Returns
        -------
        Dictionary of reweighted injections
        '''
        prob = jnp.exp(self.log_weights)
        prob/=prob.sum()
        idx = jnp.random.choice(len(self.prior),replace=replace,p=prob)
        return {key:self.injections_data[key][idx] for key in list(self.injections_data.keys())}
        
        
        
 