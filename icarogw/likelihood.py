from .cupy_pal import *
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
        
        
        # Saves the minimum number of samples to use per event
        nsamps=np.array([posterior_samples_dict[key].nsamples for key in posterior_samples_dict.keys()])
        if nparallel is None:
            nparallel=np.min(nsamps)
        else:
            nparallel=np.min(np.hstack([nsamps,nparallel]))
        self.nparallel=np2cp(nparallel)
            
        n_ev=len(posterior_samples_dict)
        llev=list(posterior_samples_dict.keys())
        print('Using {:d} samples from each {:d} posteriors'.format(self.nparallel,n_ev))
        self.n_ev=n_ev
        if neffINJ is None:
            self.neffINJ=4*self.n_ev
        else:
            self.neffINJ=neffINJ
        self.posterior_parallel={key:xp.empty([n_ev,nparallel],
                                              dtype=posterior_samples_dict[llev[0]].posterior_data[key].dtype) for key in self.rate_model.GW_parameters+['prior']}

        # Saves the posterior samples in a dictionary containing events on rows and posterior samples on columns
        for i,event in enumerate(list(posterior_samples_dict.keys())):
            posterior_samples_dict[event].cupyfy()
            len_single = posterior_samples_dict[event].nsamples
            rand_perm = xp.random.permutation(len_single)
            for key in self.posterior_parallel.keys():
                self.posterior_parallel[key][i,:]=posterior_samples_dict[event].posterior_data[key][rand_perm[:self.nparallel]]
            posterior_samples_dict[event].numpyfy()
                
        super().__init__(parameters={ll: None for ll in self.rate_model.population_parameters})
                
    def log_likelihood(self):
        '''
        Evaluates and return the log-likelihood
        '''
        
        self.rate_model.update(**{key:self.parameters[key] for key in self.rate_model.population_parameters})
        # Update the sensitivity estimation with the new model
        self.injections.update_weights(self.rate_model)            
        Neff=self.injections.effective_detection_number(self.injections.weights)
        # If the injections are not enough return 0, you cannot go to that point. This is done because the number of injections that you have
        # are not enough to calculate the selection effect
        if (Neff<self.neffINJ) | (Neff==0.):
            return float(xp.nan_to_num(-xp.inf))
            
        integ=xp.exp(self.rate_model.log_rate_PE(self.posterior_parallel['prior'],**{key:self.posterior_parallel[key] for key in self.rate_model.GW_parameters}))
        
        # Check for the number of effective sample (Eq. 2.58-2.59 document)
        sum_weights=xp.sum(integ,axis=1)/self.nparallel
        sum_weights_squared=xp.sum(xp.power(integ/self.nparallel,2.),axis=1)
        Neff_vect=xp.power(sum_weights,2.)/sum_weights_squared        
        Neff_vect[xp.isnan(Neff_vect)]=0.
        if xp.any(Neff_vect<self.neffPE):
            return float(xp.nan_to_num(-xp.inf))

        # Combine all the terms  
        if self.rate_model.scale_free:
            log_likeli = xp.sum(xp.log(sum_weights))-self.n_ev*xp.log(self.injections.pseudo_rate)
        else:
            Nexp=self.injections.expected_number_detections()
            log_likeli = -Nexp + self.n_ev*xp.log(self.injections.Tobs)+xp.sum(xp.log(sum_weights))
        
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == xp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if xp.isnan(log_likeli):
            log_likeli = float(xp.nan_to_num(-xp.inf))
        else:
            log_likeli = float(xp.nan_to_num(log_likeli))
            
        return log_likeli
                
