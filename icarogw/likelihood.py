from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn, enable_cupy
from .stochastic import spectral_siren_vanilla_omega_gw
import time
import copy
import bilby
import icarogw
from .wrappers import FlatLambdaCDM_wrap

# LVK Reviewed
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

        # enable_cupy()
        # self.injections.cupyfy()
        # self.posterior_samples_dict.cupyfy()
        # self.rate_model.cw=FlatLambdaCDM_wrap(2.)
                
        #Update the rate model with the population parameters
        self.rate_model.update(**{key:self.parameters[key] for key in self.rate_model.population_parameters})
        # Update the sensitivity estimation with the new model
        self.injections.update_weights(self.rate_model)
        Neff=self.injections.effective_injections_number()
        # If the injections are not enough return 0, you cannot go to that point. This is done because the number of injections that you have
        # are not enough to calculate the selection effect
        
        xp = get_module_array(self.injections.log_weights)
        
        if (Neff<self.neffINJ) | (Neff==0.):
            return float(xp.nan_to_num(-xp.inf))
        
        # Update the weights on the PE
        self.posterior_samples_dict.update_weights(self.rate_model)
        if xp.any(self.posterior_samples_dict.get_effective_number_of_PE()<self.neffPE):
            return float(xp.nan_to_num(-xp.inf))
        
        integ=self.posterior_samples_dict.log_weights # Extract a matrix N_ev X N_samples of log weights
             
        # Combine all the terms  
        if self.rate_model.scale_free:
            # Log likelihood for scale free model, Eq. 1.3 on the document
            log_likeli = xp.sum(xp.log(self.posterior_samples_dict.sum_weights))-self.posterior_samples_dict.n_ev*xp.log(self.injections.pseudo_rate)
        else:
            Nexp=self.injections.expected_number_detections()
            # Log likelihood for  the model, Eq. 1.1 on the document
            log_likeli = -Nexp + self.posterior_samples_dict.n_ev*xp.log(self.injections.Tobs)+xp.sum(xp.log(self.posterior_samples_dict.sum_weights))
        
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == xp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if xp.isnan(log_likeli):
            log_likeli = float(xp.nan_to_num(-xp.inf))
        else:
            log_likeli = float(xp.nan_to_num(log_likeli))
            
        return float(cp2np(log_likeli))
                

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
        
        xp = get_module_array(self.injections.log_weights)

        Nexp=self.injections.expected_number_detections()
        # Log likelihood for  the model, Eq. 1.1 on the document
        log_likeli = -Nexp 
        
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == xp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if xp.isnan(log_likeli):
            log_likeli = float(xp.nan_to_num(-xp.inf))
        else:
            log_likeli = float(xp.nan_to_num(log_likeli))
            
        return float(cp2np(log_likeli))


class Poisson_times_Stochastic_CBC_likelihood(hierarchical_likelihood):
    def __init__(self, posterior_samples_dict, injections, rate_model, freqs, look_up_Om0, stochastic_data, nparallel=None, neffPE=20, neffINJ=None):
        super().__init__(posterior_samples_dict, injections, rate_model, nparallel, neffPE, neffINJ)
        self.freqs = freqs
        self.look_up_Om0 = look_up_Om0
        self.stochastic_data = stochastic_data
    def log_likelihood(self):
        hierarchical_log_likelihood = super().log_likelihood()
        stochastic_log_likelihood = self.stochastic_log_likelihood()
        return hierarchical_log_likelihood + stochastic_log_likelihood
    def stochastic_log_likelihood(self):
        Cf = self.stochastic_data['Cf']
        sigma2s = self.stochastic_data['sigma2s']
        # Rate model is updated from the call of the poisson likelihood
        omega_gw = spectral_siren_vanilla_omega_gw(self.freqs, self.look_up_Om0, self.rate_model)

        # likelihood in eq. 23
        diff = omega_gw - Cf
        log_stoch = -0.5 * np.sum(((diff**2.) / sigma2s))

        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python value 1e-309
        log_stoch = np.nan_to_num(log_stoch, nan=-np.inf)
        
        return float(cp2np(log_stoch))

    
