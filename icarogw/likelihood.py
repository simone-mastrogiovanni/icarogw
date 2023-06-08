from .jax_pal import *
from .conversions import *
from .wrappers import *
import copy
import bilby

class hierarchical_likelihood(bilby.Likelihood):
    def __init__(self, posterior_samples_dict, injections, rate_model, nparallel=None, neffPE=20,neffINJ=None):
        '''
        Base class for an hierachical liklihood. It just saves all the input requirements for a general hierarchical analysis
        
        Parameters
        ----------
        posterior_samples_dict: dict
            Dictionary containing the posterior samples class
        injections: class
            Injection class from its module 
        rate_model: class
            Rate model to compute the CBC rate per year at the detector, taken from the wrapper module.
        nparallel: int
            Number of samples to use per event, if None it will use the maximum number of PE samples in common to all events
        neffPE: int
            Effective number of samples per event that must contribute the prior evaluation
        neffINJ: int
            Number of effective injections needed to evaluate the selection bias, if None we will assume 4* observed signals.
        '''
        
        # Saves injections in a cupyfied format
        self.injections=injections
        self.injections.cupyfy()
        self.neffPE=neffPE
        self.rate_model=rate_model
        self.posterior_samples_dict=posterior_samples_dict
        
        self.posterior_samples_dict.build_parallel_posterior(nparallel=nparallel)
        
        if neffINJ is None:
            self.neffINJ=4*self.posterior_samples_dict.n_ev
        else:
            self.neffINJ=neffINJ
            
        super().__init__(parameters={ll: None for ll in self.rate_model.population_parameters})
                
    def log_likelihood(self):
        '''
        Evaluates and return the log-likelihood
        '''
        
        #Update the rate model with the population parameters
        self.rate_model.update(**{key:self.parameters[key] for key in self.rate_model.population_parameters})
        # Update the sensitivity estimation with the new model
        self.injections.update_weights(self.rate_model)
        Neff=self.injections.effective_injections_number()
        # If the injections are not enough return 0, you cannot go to that point. This is done because the number of injections that you have
        # are not enough to calculate the selection effect
        if (Neff<self.neffINJ) | (Neff==0.):
            return float(jnp.nan_to_num(-jnp.inf))
        
        # Update the weights on the PE
        self.posterior_samples_dict.update_weights(self.rate_model)
        if jnp.any(self.posterior_samples_dict.get_effective_number_of_PE()<self.neffPE):
            return float(jnp.nan_to_num(-jnp.inf))
        
        integ=self.posterior_samples_dict.log_weights # Extract a matrix N_ev X N_samples of log weights
             
        # Combine all the terms  
        if self.rate_model.scale_free:
            # Log likelihood for scale free model, Eq. 1.3 on the document
            log_likeli = jnp.sum(jnp.log(self.posterior_samples_dict.sum_weights))-self.posterior_samples_dict.n_ev*jnp.log(self.injections.pseudo_rate)
        else:
            Nexp=self.injections.expected_number_detections()
            # Log likelihood for  the model, Eq. 1.1 on the document
            log_likeli = -Nexp + self.posterior_samples_dict.n_ev*jnp.log(self.injections.Tobs)+jnp.sum(jnp.log(self.posterior_samples_dict.sum_weights))
        
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == jnp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if jnp.isnan(log_likeli):
            log_likeli = float(jnp.nan_to_num(-jnp.inf))
        else:
            log_likeli = float(jnp.nan_to_num(log_likeli))
            
        return float(jnp2onp(log_likeli))
                

class hierarchical_likelihood_noevents(bilby.Likelihood):
    def __init__(self, injections, rate_model):
        '''
        Base class for an hierachical liklihood. It just saves all the input requirements for a general hierarchical analysis
        
        Parameters
        ----------
        posterior_samples_dict: dict
            Dictionary containing the posterior samples class
        injections: class
            Injection class from its module 
        rate_model: class
            Rate model to compute the CBC rate per year at the detector, taken from the wrapper module.
        '''
        
        # Saves injections in a cupyfied format
        self.injections=injections
        self.injections.cupyfy()
        self.rate_model=rate_model
        super().__init__(parameters={ll: None for ll in self.rate_model.population_parameters})
                
    def log_likelihood(self):
        '''
        Evaluates and return the log-likelihood
        '''
        #Update the rate model with the population parameters
        self.rate_model.update(**{key:self.parameters[key] for key in self.rate_model.population_parameters})
        # Update the sensitivity estimation with the new model
        self.injections.update_weights(self.rate_model)
        
        Nexp=self.injections.expected_number_detections()
        # Log likelihood for  the model, Eq. 1.1 on the document
        log_likeli = -Nexp 
        
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == jnp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if jnp.isnan(log_likeli):
            log_likeli = float(jnp.nan_to_num(-jnp.inf))
        else:
            log_likeli = float(jnp.nan_to_num(log_likeli))
            
        return float(jnp2onp(log_likeli))
                
