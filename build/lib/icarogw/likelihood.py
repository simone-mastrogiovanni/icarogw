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
                
                
class hierarchical_likelihood_CBC_source_counterpart(hierarchical_likelihood):
    def __init__(self, posterior_samples_dict, injections, cosmology_wrap, mass_wrap, rate_wrap, spin_wrap = None,
                 nparallel=None, neffPE=20,neffINJ=None):
        '''
        This is a class containing the hierarchical likelihood for the CBC in source frame when there is an EM counterpart.
        This likelihood class is available only in scale-free mode.
        
        Parameters
        ----------
    
        posterior_samples_dict: dict
            Dictionary containing the posterior samples class. Posterior samples must be initialized with the EM counterpart.
        injections: class
            Injection class from its module 
        class: class
            Galaxy catalog class
        cosmology_wrap, mass_wrap,rate_wrap, spin_wrap: classes
            The cosmology, mass, rate and spin wrappers to use
        nparallel: int
            Number of samples to use per event, if None it will use the maximum number of PE samples in common to all events
        neffPE: int
            Effective number of samples per event that must contribute the prior evaluation
        neffINJ: int
            Number of effective injections needed to evaluate the selection bias, if None we will assume 4* observed signals.
        '''
        
        self.z_em_list = []
        
        for key in posterior_samples_dict.keys():
            posterior_samples_dict[key].cupyfy()
            self.z_em_list.append(posterior_samples_dict[key].z_EM)
            
        # Initialize the data arrays that you need
        super().__init__(posterior_samples_dict, injections, nparallel=nparallel,neffPE=neffPE,neffINJ=neffINJ)
        
        self.cosmology_wrap=cosmology_wrap
        self.mass_wrap=mass_wrap
        self.rate_wrap=rate_wrap
        self.spin_wrap=spin_wrap
                    
        self.tot_list=self.mass_wrap.parameters+self.rate_wrap.parameters+self.cosmology_wrap.parameters
            
        if self.spin_wrap is not None:
            self.tot_list=self.tot_list+self.spin_wrap.parameters
      
        # Initialize the Bilby parameters
        super(hierarchical_likelihood,self).__init__(parameters={ll: None for ll in self.tot_list})
               
    def log_likelihood(self):
        '''
        Evaluates and return the log-likelihood
        '''
 
        self.cosmology_wrap.update(**{key:self.parameters[key] for key in self.cosmology_wrap.parameters})
        self.mass_wrap.update(**{key:self.parameters[key] for key in self.mass_wrap.parameters})
        self.rate_wrap.update(**{key:self.parameters[key] for key in self.rate_wrap.parameters})
        if self.spin_wrap is not None:
            self.spin_wrap.update(**{key:self.parameters[key] for key in self.spin_wrap.parameters})
        
        # Update the sensitivity estimation with the new model
        self.injections.update_VT(self.cosmology_wrap.cosmology,self.mass_wrap,self.rate_wrap,spin_wrap=self.spin_wrap)
                
        Neff=self.injections.effective_detection_number(self.injections.weights)
        # If the injections are not enough return 0, you cannot go to that point. This is done because the number of injections that you have
        # are not enough to calculate the selection effect
        if (Neff<self.neffINJ) | (Neff==0.):
            return float(xp.nan_to_num(-xp.inf))
        
        ms1, ms2, z = detector2source(self.posterior_parallel['mass_1'],self.posterior_parallel['mass_2']
                                              ,self.posterior_parallel['luminosity_distance'],self.cosmology_wrap.cosmology)
        
        dVc_dz=self.cosmology_wrap.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi 
                
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        weights=(self.mass_wrap.pdf(ms1,ms2)*self.rate_wrap.rate.evaluate(z)*dVc_dz)/(
            (1.+z)*detector2source_jacobian(z,self.cosmology_wrap.cosmology)*self.posterior_parallel['prior']*self.nparallel)
        
        if self.spin_wrap is not None:
            weights*=self.spin_wrap.pdf(**{key:self.posterior_parallel[key] for key in self.spin_wrap.input_eval_parameters})
            
        weights=xp.reshape(weights,self.origin_shape)
        
        # Check for the number of effective sample (Eq. 2.58-2.59 document)
        sum_weights=xp.sum(weights,axis=1)
        sum_weights_squared=xp.sum(xp.power(weights,2.),axis=1)
        Neff_vect=xp.power(sum_weights,2.)/sum_weights_squared        
        Neff_vect[xp.isnan(Neff_vect)]=0.
        if xp.any(Neff_vect<self.neffPE):
            return float(xp.nan_to_num(-xp.inf))
        
        log_l = xp.empty(self.n_ev)
        
        ms1, ms2, z = xp.reshape(ms1,self.origin_shape), xp.reshape(ms2,self.origin_shape), xp.reshape(z,self.origin_shape)
        
        for i in range(self.n_ev): 
            kde_fit = gaussian_kde(z[i,:],weights=weights[i,:])
            log_l[i] = np.log(sum_weights[i])+np.log(kde_fit(self.z_em_list[i]).sum()/len(self.z_em_list[i]))-xp.log(self.injections.VT)
        
        log_likeli = xp.sum(log_l)

        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == xp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if xp.isnan(log_likeli):
            log_likeli = float(xp.nan_to_num(-xp.inf))
        else:
            log_likeli = float(xp.nan_to_num(log_likeli))
            
        return log_likeli
                
class hierarchical_likelihood_CBC_source(hierarchical_likelihood):
    def __init__(self, posterior_samples_dict, injections, cosmology_wrap, mass_wrap, rate_wrap, spin_wrap = None,
                 nparallel=None, scale_free=False, neffPE=20,neffINJ=None):
        '''
        This is a class containing the hierarchical likelihood for the CBC in source frame.
        
        Parameters
        ----------
    
        posterior_samples_dict: dict
            Dictionary containing the posterior samples class
        injections: class
            Injection class from its module 
        class: class
            Galaxy catalog class
        cosmology_wrap, mass_wrap,rate_wrap, spin_wrap: classes
            The cosmology, mass, rate and spin wrappers to use
        nparallel: int
            Number of samples to use per event, if None it will use the maximum number of PE samples in common to all events
        neffPE: int
            Effective number of samples per event that must contribute the prior evaluation
        neffINJ: int
            Number of effective injections needed to evaluate the selection bias, if None we will assume 4* observed signals.
        '''
        
        self.scale_free=scale_free
                
        for key in posterior_samples_dict.keys():
            posterior_samples_dict[key].cupyfy()
            
        # Initialize the data arrays that you need
        super().__init__(posterior_samples_dict, injections, nparallel=nparallel,neffPE=neffPE,neffINJ=neffINJ)
        
        self.cosmology_wrap=cosmology_wrap
        self.mass_wrap=mass_wrap
        self.rate_wrap=rate_wrap
        self.spin_wrap=spin_wrap
                    
        if scale_free:
            self.tot_list=self.mass_wrap.parameters+self.rate_wrap.parameters+self.cosmology_wrap.parameters
        else:
            self.tot_list=self.mass_wrap.parameters+self.rate_wrap.parameters+self.cosmology_wrap.parameters+['R0']
            
        if self.spin_wrap is not None:
            self.tot_list=self.tot_list+self.spin_wrap.parameters
  
            
        # Initialize the Bilby parameters
        super(hierarchical_likelihood,self).__init__(parameters={ll: None for ll in self.tot_list})
               
    def log_likelihood(self):
        '''
        Evaluates and return the log-likelihood
        '''
 
        self.cosmology_wrap.update(**{key:self.parameters[key] for key in self.cosmology_wrap.parameters})
        self.mass_wrap.update(**{key:self.parameters[key] for key in self.mass_wrap.parameters})
        self.rate_wrap.update(**{key:self.parameters[key] for key in self.rate_wrap.parameters})
        if self.spin_wrap is not None:
            self.spin_wrap.update(**{key:self.parameters[key] for key in self.spin_wrap.parameters})
        
        # Update the sensitivity estimation with the new model
        self.injections.update_VT(self.cosmology_wrap.cosmology,self.mass_wrap,self.rate_wrap,spin_wrap=self.spin_wrap)
                
        Neff=self.injections.effective_detection_number(self.injections.weights)
        # If the injections are not enough return 0, you cannot go to that point. This is done because the number of injections that you have
        # are not enough to calculate the selection effect
        if (Neff<self.neffINJ) | (Neff==0.):
            return float(xp.nan_to_num(-xp.inf))
        

        ms1, ms2, z = detector2source(self.posterior_parallel['mass_1'],self.posterior_parallel['mass_2']
                                              ,self.posterior_parallel['luminosity_distance'],self.cosmology_wrap.cosmology)
        
        
        dVc_dz=self.cosmology_wrap.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi 
                
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        integ=(self.mass_wrap.pdf(ms1,ms2)*self.rate_wrap.rate.evaluate(z)*dVc_dz)/(
            (1.+z)*detector2source_jacobian(z,self.cosmology_wrap.cosmology)*self.posterior_parallel['prior']*self.nparallel)
        
        if self.spin_wrap is not None:
            integ*=self.spin_wrap.pdf(**{key:self.posterior_parallel[key] for key in self.spin_wrap.input_eval_parameters})
        
        
        integ=xp.reshape(integ,self.origin_shape)
        
        # Check for the number of effective sample (Eq. 2.58-2.59 document)
        sum_weights=xp.sum(integ,axis=1)
        sum_weights_squared=xp.sum(xp.power(integ,2.),axis=1)
        Neff_vect=xp.power(sum_weights,2.)/sum_weights_squared        
        Neff_vect[xp.isnan(Neff_vect)]=0.
        if xp.any(Neff_vect<self.neffPE):
            return float(xp.nan_to_num(-xp.inf))

        # Combine all the terms  
        if self.scale_free:
            log_likeli = xp.sum(xp.log(sum_weights))-self.n_ev*xp.log(self.injections.VT)
        else:
            Nexp=self.injections.expected_number_detection(self.parameters['R0'])
            log_likeli = -Nexp + self.n_ev*xp.log(self.parameters['R0']*self.injections.Tobs)+xp.sum(xp.log(sum_weights))
        
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == xp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if xp.isnan(log_likeli):
            log_likeli = float(xp.nan_to_num(-xp.inf))
        else:
            log_likeli = float(xp.nan_to_num(log_likeli))
            
        return log_likeli
    
                
class hierarchical_likelihood_galaxies(hierarchical_likelihood):
    def __init__(self, posterior_samples_dict, injections, catalog, cosmology_wrap, mass_wrap, rate_wrap, spin_wrap=None,
                 nparallel=None, scale_free=False, neffPE=20,neffINJ=None,average=True):
        '''
        This is a class containing the hierarchical likelihood for the galaxy catalog case, Rgal method.
        
        Parameters
        ----------
    
        posterior_samples_dict: dict
            Dictionary containing the posterior samples class
        injections: class
            Injection class from its module 
        class: class
            Galaxy catalog class
        cosmology_wrap, mass_wrap,rate_wrap, spin_wrap: classes
            The cosmology, mass, rate and spin wrappers to use
        nparallel: int
            Number of samples to use per event, if None it will use the maximum number of PE samples in common to all events
        average: bool
            True if you assume detection capabilities does not depend a lot on sky
        scale_free: bool
            True if you want to use the scale free version of the prior
        neffPE: int
            Effective number of samples per event that must contribute the prior evaluation
        neffINJ: int
            Number of effective injections needed to evaluate the selection bias, if None we will assume 4* observed signals.
        average: bool
            If true it will average the detection probability over the entire sky.
        '''
        self.scale_free=scale_free
        self.catalog=catalog
        self.average=average
        
        # Check that nside of the catalog is the same of injections
        if injections.nside!=catalog.hdf5pointer['catalog'].attrs['nside']:
            raise ValueError('You must use the same nside between Catalog and injections')
        
        print('Pixelizing the posterior samples with nside={:d}'.format(catalog.hdf5pointer['catalog'].attrs['nside']))
        for key in posterior_samples_dict.keys():
            posterior_samples_dict[key].numpyfy()
            posterior_samples_dict[key].pixelize(catalog.hdf5pointer['catalog'].attrs['nside'])
            posterior_samples_dict[key].cupyfy()
            
        # Initialize the data arrays that you need
        super().__init__(posterior_samples_dict, injections, nparallel=nparallel,neffPE=neffPE,neffINJ=neffINJ)
    
        self.posterior_parallel['sky_indices']=self.posterior_parallel['sky_indices'].astype(int)
        
        self.cosmology_wrap=cosmology_wrap
        self.mass_wrap=mass_wrap
        self.rate_wrap=rate_wrap
        self.spin_wrap=spin_wrap
            
        if scale_free:
            self.tot_list=self.mass_wrap.parameters+self.rate_wrap.parameters+self.cosmology_wrap.parameters
        else:
            self.tot_list=self.mass_wrap.parameters+self.rate_wrap.parameters+self.cosmology_wrap.parameters+['Rgal']
            
        if self.spin_wrap is not None:
            self.tot_list=self.tot_list+self.spin_wrap.parameters
            
        # Initialize the Bilby parameters
        super(hierarchical_likelihood,self).__init__(parameters={ll: None for ll in self.tot_list})
               
    def log_likelihood(self):
        '''
        Evaluates and return the log-likelihood
        '''
 
        self.cosmology_wrap.update(**{key:self.parameters[key] for key in self.cosmology_wrap.parameters})
        self.mass_wrap.update(**{key:self.parameters[key] for key in self.mass_wrap.parameters})
        self.rate_wrap.update(**{key:self.parameters[key] for key in self.rate_wrap.parameters})
        if self.spin_wrap is not None:
            self.spin_wrap.update(**{key:self.parameters[key] for key in self.spin_wrap.parameters})
            
        self.catalog.sch_fun.build_MF(self.cosmology_wrap.cosmology)

        # Update the sensitivity estimation with the new model
        self.injections.update_galaxies(self.catalog,self.cosmology_wrap.cosmology,self.mass_wrap,self.rate_wrap,spin_wrap=self.spin_wrap,average=self.average)
                
        Neff=self.injections.effective_detection_number(self.injections.weights_gal)
        # If the injections are not enough return 0, you cannot go to that point. This is done because the number of injections that you have
        # are not enough to calculate the selection effect
        if (Neff<self.neffINJ) | (Neff==0.):
            return float(xp.nan_to_num(-xp.inf))
        

        ms1, ms2, z = detector2source(self.posterior_parallel['mass_1'],self.posterior_parallel['mass_2']
                                              ,self.posterior_parallel['luminosity_distance'],self.cosmology_wrap.cosmology)
        
        dNgal_cat,dNgal_bg=self.catalog.effective_galaxy_number_interpolant(z,self.posterior_parallel['sky_indices'],self.cosmology_wrap.cosmology
                                                    ,dl=self.posterior_parallel['luminosity_distance'])

        # Effective number density of galaxies (Eq. 2.19 on the overleaf document)
        dNgaleff=dNgal_cat+dNgal_bg
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        integ=(self.mass_wrap.pdf(ms1,ms2)*(0.25/xp.pi)*self.rate_wrap.rate.evaluate(z)*dNgaleff)/(
            (1.+z)*detector2source_jacobian(z,self.cosmology_wrap.cosmology)*self.posterior_parallel['prior']*self.nparallel)
        
        if self.spin_wrap is not None:
            integ*=self.spin_wrap.pdf(**{key:self.posterior_parallel[key] for key in self.spin_wrap.input_eval_parameters})
        
        integ=xp.reshape(integ,self.origin_shape)
        
        # Check for the number of effective sample (Eq. 2.58-2.59 document)
        sum_weights=xp.sum(integ,axis=1)
        sum_weights_squared=xp.sum(xp.power(integ,2.),axis=1)
        Neff_vect=xp.power(sum_weights,2.)/sum_weights_squared        
        Neff_vect[xp.isnan(Neff_vect)]=0.
        if xp.any(Neff_vect<self.neffPE):
            return float(xp.nan_to_num(-xp.inf))

        # Combine all the terms  
        if self.scale_free:
            log_likeli = xp.sum(xp.log(sum_weights))-self.n_ev*xp.log(self.injections.dNgal)
        else:
            Nexp=self.injections.expected_number_detection_galaxies(self.parameters['Rgal'])
            log_likeli = -Nexp + self.n_ev*xp.log(self.parameters['Rgal']*self.injections.Tobs)+xp.sum(xp.log(sum_weights))
        
        # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
        # python valye 1e-309
        if log_likeli == xp.inf:
            raise ValueError('LOG-likelihood must be smaller than infinite')

        if xp.isnan(log_likeli):
            log_likeli = float(xp.nan_to_num(-xp.inf))
        else:
            log_likeli = float(xp.nan_to_num(log_likeli))
            
        return log_likeli

