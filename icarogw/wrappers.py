from .cupy_pal import *
from .cosmology import *
from .conversions import detector2source_jacobian, detector2source
from .priors import *
from scipy.stats import gaussian_kde
import copy

from astropy.cosmology import FlatLambdaCDM, FlatwCDM

################ BEGIN: Wrappers to compute the CBC rate per year at the detector below ###############

class CBC_catalog_vanilla_rate_skymap(object):
    
    def __init__(self,catalog,cosmology_wrapper,rate_wrapper, average=False,scale_free=False):
        '''
        A wrapper for the CBC rate model that make use of of just the luminosity distance and position of GW events.
        Useful if you generate PE from skymaps

        Parameters
        ----------
        catalog: object
            icarogw catalog class containing the preprocessed galaxy catalog
        cosmology_wrapper: object
            Cosmology wrapper from icarogw
        rate_wrapper: object
            Merger rate wrapper from icarogw
        '''
        
        self.catalog = catalog
        self.cw = cosmology_wrapper
        self.rw = rate_wrapper
        self.average = average
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters =  self.cw.population_parameters+self.rw.population_parameters
        else:
            self.population_parameters =  self.cw.population_parameters+self.rw.population_parameters + ['Rgal']
            
        event_parameters = ['luminosity_distance','sky_indices']
        
        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
            
        if not self.scale_free:
            self.Rgal = kwargs['Rgal']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        z = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        dNgal_cat,dNgal_bg=self.catalog.effective_galaxy_number_interpolant(z,kwargs['sky_indices'],self.cw.cosmology
                                                    ,dl=kwargs['luminosity_distance'],average=False)

        # Effective number density of galaxies (Eq. 2.19 on the overleaf document)
        dNgaleff=dNgal_cat+dNgal_bg
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.rw.rate.log_evaluate(z)+xp.log(dNgaleff) \
        -xp.log1p(z)-xp.log(xp.abs(self.cw.cosmology.ddl_by_dz_at_z(z)))-xp.log(prior)
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.Rgal)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        z = self.cw.cosmology.dl2z(kwargs['luminosity_distance'])
        dNgal_cat,dNgal_bg=self.catalog.effective_galaxy_number_interpolant(z,kwargs['sky_indices'],self.cw.cosmology
                                                    ,dl=kwargs['luminosity_distance'],average=self.average)

        # Effective number density of galaxies (Eq. 2.19 on the overleaf document)
        dNgaleff=dNgal_cat+dNgal_bg
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.rw.rate.log_evaluate(z)+xp.log(dNgaleff) \
        -xp.log1p(z)-xp.log(xp.abs(self.cw.cosmology.ddl_by_dz_at_z(z)))-xp.log(prior)
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.Rgal)
        else:
            log_out = log_weights
            
        return log_out

class CBC_low_latency_skymap_EM_counterpart(object):
    def __init__(self,cosmology_wrapper,rate_wrapper, list_of_skymaps,scale_free=False):

        '''
        A wrapper for the CBC rate model that make use of LVK low latency skymaps and EM counterparts.
        Posterior samples are going to be possible EM counterparts (1 for each event). These EM PEs
        are then combined with the GW skymap

        Parameters
        ----------
        cosmology_wrapper: object
            Cosmology wrapper from icarogw
        rate_wrapper: object
            Merger rate wrapper from icarogw
        list_of_skymaps: object
            A list of icarogw skymaps objects, order must be compatible with catalog of EM counterparts passed in posterior sampels
        scale_free: True
            Scale free model or not
        '''
        
        self.cw = cosmology_wrapper
        self.rw = rate_wrapper
        self.scale_free = scale_free
        self.list_of_skymaps=list_of_skymaps
         
        if scale_free:
            self.population_parameters =  self.cw.population_parameters+self.rw.population_parameters
        else:
            self.population_parameters =  self.cw.population_parameters+self.rw.population_parameters + ['R0']
            
        self.PEs_parameters = ['z_EM','right_ascension','declination']
        self.injections_parameters = ['luminosity_distance']

        
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        if len(kwargs['z_EM'].shape) != 2:
            raise ValueError('The EM counterpart rate wants N_ev x N_samples arrays')

        dl_samples = self.cw.cosmology.z2dl(kwargs['z_EM'])       
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(kwargs['z_EM'])*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.rw.rate.log_evaluate(kwargs['z_EM'])+log_dVc_dz-xp.log(prior)-xp.log1p(kwargs['z_EM'])       
        
        n_ev = kwargs['z_EM'].shape[0]
        lwtot = xp.empty(kwargs['z_EM'].shape)
        for i in range(n_ev): 
            self.list_of_skymaps[i].intersect_EM_PE(kwargs['right_ascension'][i,:],kwargs['declination'][i,:])
            log_l_skymap = xp.log(self.list_of_skymaps[i].evaluate_3D_likelihood_intersected(dl_samples[i,:]))            
            lwtot[i,:] = logsumexp(log_weights[i,:]+log_l_skymap)-xp.log(kwargs['z_EM'].shape[1])

        if not self.scale_free:
            log_out = lwtot + xp.log(self.R0)
        else:
            log_out = lwtot
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        z = self.cw.cosmology.dl2z(kwargs['luminosity_distance']) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(xp.abs(self.cw.cosmology.ddl_by_dz_at_z(z)))-xp.log1p(z)
        
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out


class CBC_vanilla_EM_counterpart(object):
    '''
    This is a rate model that parametrizes the CBC rate per year at the detector in terms of source-frame
    masses, spin parameters and redshift rate evolution times differential of comoving volume. Source-frame mass distribution,
    spin distribution and redshift distribution are summed to be independent from each other.

    .. math::
        \\frac{d N_{\\rm CBC}(\\Lambda)}{d\\vec{m}d\\vec{\\chi} dz dt_s} = R_0 \\psi(z;\\Lambda) p_{\\rm pop}(\\vec{m},\\vec{\\chi}|\\Lambda) \\frac{d V_c}{dz}

    The wrapper works with luminosity distances and detector frame masses and optionally with some chosen spin parameters.

    Note that this rate model also takes into account for the GW events an additional weight given by the EM counterpart, we defer to section 2.3 for more details.
    Note also that for evaluating selection biases, we do not account for biases given by EM observatories.

    Parameters
    ----------
    cosmology_wrapper: class
        Wrapper for the cosmological model
    mass_wrapper: class
        Wrapper for the source-frame mass distribution
    rate_wrapper: class
        Wrapper for the rate evolution model
    spin_wrapper: class
        Wrapper for the rate model.
    scale_free: bool
        True if you want to use the model for scale-free likelihood (no R0)
    '''
    def __init__(self,cosmology_wrapper,mass_wrapper,rate_wrapper,spin_wrapper=None,scale_free=False):
        self.cw = cosmology_wrapper
        self.mw = mass_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters
        else:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters + ['R0']

        event_parameters = ['mass_1', 'mass_2', 'luminosity_distance','z_EM']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy().remove('z_EM')
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        if len(kwargs['mass_1'].shape) != 2:
            raise ValueError('The EM counterpart rate wants N_ev x N_samples arrays')
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
        
        n_ev = kwargs['mass_1'].shape[0]
        lwtot = xp.empty(kwargs['z_EM'].shape)
        for i in range(n_ev): 
            ww = xp.exp(log_weights[i,:])
            kde_fit = gaussian_kde(z[i,:],weights=ww/ww.sum())   
            lwtot[i,:] = logsumexp(log_weights[i,:])-xp.log(kwargs['mass_1'].shape[1])+kde_fit.logpdf(kwargs['z_EM'][i,:])

        if not self.scale_free:
            log_out = lwtot + xp.log(self.R0)
        else:
            log_out = lwtot
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out


class CBC_vanilla_rate(object):
    '''
    This is a rate model that parametrizes the CBC rate per year at the detector in terms of source-frame
    masses, spin parameters and redshift rate evolution times differential of comoving volume. Source-frame mass distribution,
    spin distribution and redshift distribution are summed to be independent from each other.

    .. math::
        \\frac{d N_{\\rm CBC}(\\Lambda)}{d\\vec{m}d\\vec{\\chi} dz dt_s} = R_0 \\psi(z;\\Lambda) p_{\\rm pop}(\\vec{m},\\vec{\\chi}|\\Lambda) \\frac{d V_c}{dz}

    The wrapper works with luminosity distances and detector frame masses and optionally with some chosen spin parameters, used to compute the rate.

    Parameters
    ----------
    cosmology_wrapper: class
        Wrapper for the cosmological model
    mass_wrapper: class
        Wrapper for the source-frame mass distribution
    rate_wrapper: class
        Wrapper for the rate evolution model
    spin_wrapper: class
        Wrapper for the rate model.
    scale_free: bool
        True if you want to use the model for scale-free likelihood (no R0)
    '''
    def __init__(self,cosmology_wrapper,mass_wrapper,rate_wrapper,spin_wrapper=None,scale_free=False):
        
        self.cw = cosmology_wrapper
        self.mw = mass_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters
        else:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters + ['R0']
            
        event_parameters = ['mass_1', 'mass_2', 'luminosity_distance']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out
    
class CBC_catalog_vanilla_rate(object):
    '''
    This is a rate model that parametrizes the CBC rate per year at the detector in terms of source-frame
    masses, spin parameters and redshift rate evolution times galaxy number density. This rate model also uses galaxy catalogs.
    Source-frame mass distribution, spin distribution and redshift distribution are summed to be independent from each other.

    .. math::
        \\frac{dN_{\\rm CBC}(\\Lambda)}{dz d\\vec{m} d\\vec{\\chi} d\\Omega dt_s} = R^{*}_{\\rm gal,0} \\psi(z;\\Lambda) p_{\\rm pop}(\\vec{m},  \\vec{\\chi}|\\Lambda) \\times 
        
    .. math::
        \\times \\left[ \\frac{dV_c}{dz d\\Omega} \\phi_*(H_0)\\Gamma_{\\rm inc}(\\alpha+\\epsilon+1,x_{\\rm max}(M_{\\rm thr}),x_{\\rm min}) + \\sum_{i=1}^{N_{\\rm gal}(\\Omega)} f_{L}(M(m_i,z);\\Lambda) p(z|z^i_{\\rm obs},\\sigma^i_{\\rm z,obs}) \\right],

    The wrapper works with luminosity distances, detector frame masses and sky pixels and optionally with some chosen spin parameters.

    Parameters
    ----------
    catalog: class
        Catalog class already processed to caclulate selection biases from the galaxy catalog.
    cosmology_wrapper: class
        Wrapper for the cosmological model
    mass_wrapper: class
        Wrapper for the source-frame mass distribution
    rate_wrapper: class
        Wrapper for the rate evolution model
    spin_wrapper: class
        Wrapper for the rate model.
    scale_free: bool
        True if you want to use the model for scale-free likelihood (no R0)
    '''
    def __init__(self,catalog,cosmology_wrapper,mass_wrapper,rate_wrapper,spin_wrapper=None, average=False,scale_free=False):
        
        self.catalog = catalog
        self.cw = cosmology_wrapper
        self.mw = mass_wrapper
        self.rw = rate_wrapper
        self.sw = spin_wrapper
        self.average = average
        self.scale_free = scale_free
        
        if scale_free:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters
        else:
            self.population_parameters =  self.cw.population_parameters+self.mw.population_parameters+self.rw.population_parameters + ['Rgal']
            
        event_parameters = ['mass_1', 'mass_2', 'luminosity_distance','sky_indices']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            event_parameters = event_parameters + self.sw.event_parameters

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()
            
    def update(self,**kwargs):
        '''
        This method updates the population models encoded in the wrapper. 
        
        Parameters
        ----------
        kwargs: flags
            The kwargs passed should be the population parameters given in self.population_parameters
        '''
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            
            self.Rgal = kwargs['Rgal']
        
    def log_rate_PE(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the posterior samples.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology)
        dNgal_cat,dNgal_bg=self.catalog.effective_galaxy_number_interpolant(z,kwargs['sky_indices'],self.cw.cosmology
                                                    ,dl=kwargs['luminosity_distance'],average=False)

        # Effective number density of galaxies (Eq. 2.19 on the overleaf document)
        dNgaleff=dNgal_cat+dNgal_bg
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+xp.log(dNgaleff) \
        -xp.log1p(z)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log(prior)
        
        if self.sw is not None:
            log_weights+=self.spin_wrap.log_pdf(**{key:self.posterior_parallel[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.Rgal)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        '''
        This method calculates the weights (CBC merger rate per year at detector) for the injections.
        
        Parameters
        ----------
        prior: array
            Prior written in terms of the variables identified by self.event_parameters
        kwargs: flags
            The kwargs are identified by self.event_parameters. Note that if the prior is scale-free, the overall normalization will not be included.
        '''
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology)
        dNgal_cat,dNgal_bg=self.catalog.effective_galaxy_number_interpolant(z,kwargs['sky_indices'],self.cw.cosmology
                                                    ,dl=kwargs['luminosity_distance'],average=self.average)

        # Effective number density of galaxies (Eq. 2.19 on the overleaf document)
        dNgaleff=dNgal_cat+dNgal_bg
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+xp.log(dNgaleff) \
        -xp.log1p(z)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log(prior)
        
        if self.sw is not None:
            log_weights+=self.spin_wrap.log_pdf(**{key:self.posterior_parallel[key] for key in self.sw.event_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.Rgal)
        else:
            log_out = log_weights
            
        return log_out

################ END: Wrappers to compute the CBC rate per year at the detector below ###############


################ BEGIN: Small wrappers used in support of the main wrappers above ###################

def mass_wrappers_init(name):
    if name == 'PowerLaw':
        mass_wrap=massprior_PowerLaw()
    elif name == 'PowerLawPeak':
        mass_wrap=massprior_PowerLawPeak()
    elif name == 'BrokenPowerLaw':
        mass_wrap=massprior_BrokenPowerLaw()
    elif name == 'MultiPeak':
        mass_wrap=massprior_MultiPeak()
    elif name == 'PowerLaw_NSBH':
        mass_wrap=massprior_PowerLaw_NSBH()
    elif name == 'PowerLawPeak_NSBH':
        mass_wrap=massprior_PowerLawPeak_NSBH()
    elif name == 'BrokenPowerLaw_NSBH':
        mass_wrap=massprior_BrokenPowerLaw_NSBH()
    elif name == 'MultiPeak_NSBH':
        mass_wrap=massprior_MultiPeak_NSBH()    
    else:
        raise ValueError('Mass model not known') 
    return mass_wrap

def rate_wrappers_init(name):
    if name == 'PowerLaw':
        rate_wrap=rateevolution_PowerLaw()
    elif name == 'Madau':
        rate_wrap=rateevolution_Madau()
    else:
        raise ValueError('Rate model not known')    
    return rate_wrap

def cosmology_wrappers_init(name,zmax):
    if name == 'FlatLambdaCDM':
        cosmology_wrap=FlatLambdaCDM_wrap(zmax=zmax)
    elif name == 'FlatwCDM':
        cosmology_wrap=FlatwCDM_wrap(zmax=zmax)
    else:
        raise ValueError('Cosmology model not known')
    return cosmology_wrap

def modGR_wrappers_init(name,bgwrap):
    if name == 'Xi0':
        cosmology_wrap=Xi0_mod_wrap(bgwrap)
    elif name == 'cM':
        cosmology_wrap=cM_mod_wrap(bgwrap)
    elif name == 'extraD':
        cosmology_wrap=extraD_mod_wrap(bgwrap)
    elif name == 'alphalog':
        cosmology_wrap=alphalog_mod_wrap(bgwrap)
    else:
        raise ValueError('ModGR model not known')
    return cosmology_wrap

# A parent class for the standard mass probabilities
class source_mass_default(object):
    def pdf(self,mass_1_source,mass_2_source):
        return self.prior.pdf(mass_1_source,mass_2_source)
    def log_pdf(self,mass_1_source,mass_2_source):
        return self.prior.log_pdf(mass_1_source,mass_2_source)

# A parent class for the rate
class rate_default(object):
    def evaluate(self,z):
        return self.rate.evaluate(z)
    def log_evaluate(self,z):
        return self.rate.log_evaluate(z)
    

class rateevolution_PowerLaw(rate_default):
    def __init__(self):
        self.population_parameters=['gamma']
    def update(self,**kwargs):
        self.rate=powerlaw_rate(**kwargs)

class rateevolution_Madau(rate_default):
    def __init__(self):
        self.population_parameters=['gamma','kappa','zp']
    def update(self,**kwargs):
        self.rate=md_rate(**kwargs)
    
class FlatLambdaCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=FlatLambdaCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))
        
class FlatwCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0','w0']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=FlatwCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))

class Xi0_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['Xi0','n']
        self.cosmology=Xi0_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),Xi0=kwargs['Xi0'],n=kwargs['n'])

class extraD_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['D','n','Rc']
        self.cosmology=extraD_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),D=kwargs['D'],n=kwargs['n'],Rc=kwargs['Rc'])

class cM_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['cM']
        self.cosmology=cM_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),cM=kwargs['cM'])

class alphalog_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['alphalog_1','alphalog_2','alphalog_3']
        self.cosmology=alphalog_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),alphalog_1=kwargs['alphalog_1']
                                       ,alphalog_2=kwargs['alphalog_2'],alphalog_3=kwargs['alphalog_3'])
    
class massprior_PowerLaw(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha','beta','mmin','mmax']
    def update(self,**kwargs):
        p1,p2=PowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha']),PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta'])
        self.prior=conditional_2dimpdf(p1,p2)

class massprior_PowerLawPeak(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha','beta','mmin','mmax','delta_m','mu_g','sigma_g','lambda_peak']
    def update(self,**kwargs):
        p1=SmoothedProb(PowerLawGaussian(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],kwargs['lambda_peak'],kwargs['mu_g'],
                                         kwargs['sigma_g'],kwargs['mmin'],kwargs['mu_g']+5*kwargs['sigma_g']),kwargs['delta_m'])
        p2=SmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimpdf(p1,p2)
    
class massprior_BrokenPowerLaw(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','beta','mmin','mmax','delta_m','b']
    def update(self,**kwargs):
        p1=SmoothedProb(BrokenPowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b']),kwargs['delta_m'])
        p2=SmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimpdf(p1,p2)

class massprior_MultiPeak(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha','beta','mmin','mmax','delta_m','mu_g_low','sigma_g_low','lambda_g_low','mu_g_high','sigma_g_high','lambda_g']
    def update(self,**kwargs):
        p1=SmoothedProb(PowerLawTwoGaussians(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],
                                             kwargs['lambda_g'],kwargs['lambda_g_low'],kwargs['mu_g_low'],
                                             kwargs['sigma_g_low'],kwargs['mmin'],kwargs['mu_g_low']+5*kwargs['sigma_g_low'],
                                             kwargs['mu_g_high'],kwargs['sigma_g_high'],kwargs['mmin'],kwargs['mu_g_high']+5*kwargs['sigma_g_high']),kwargs['delta_m'])
        p2=SmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimpdf(p1,p2)
        
class massprior_PowerLaw_NSBH(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha','beta','mmin','mmax','mmin_NS','mmax_NS']
    def update(self,**kwargs):
        p1,p2=PowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha']),PowerLaw(kwargs['mmin_NS'],kwargs['mmax_NS'],kwargs['beta'])
        self.prior=conditional_2dimpdf(p1,p2)

class massprior_PowerLawPeak_NSBH(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha','beta','mmin','mmax','delta_m','mu_g','sigma_g','lambda_peak','mmin_NS','mmax_NS','delta_m_NS']
    def update(self,**kwargs):
        p1=SmoothedProb(PowerLawGaussian(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],kwargs['lambda_peak'],kwargs['mu_g'],
                                         kwargs['sigma_g'],kwargs['mmin'],kwargs['mu_g']+5*kwargs['sigma_g']),kwargs['delta_m'])
        p2=SmoothedProb(PowerLaw(kwargs['mmin_NS'],kwargs['mmax_NS'],kwargs['beta']),kwargs['delta_m_NS'])
        self.prior=conditional_2dimpdf(p1,p2)
    
class massprior_BrokenPowerLaw_NSBH(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','beta','mmin','mmax','delta_m','b','mmin_NS','mmax_NS','delta_m_NS']
    def update(self,**kwargs):
        p1=SmoothedProb(BrokenPowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b']),kwargs['delta_m'])
        p2=SmoothedProb(PowerLaw(kwargs['mmin_NS'],kwargs['mmax_NS'],kwargs['beta']),kwargs['delta_m_NS'])
        self.prior=conditional_2dimpdf(p1,p2)

class massprior_MultiPeak_NSBH(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha','beta','mmin','mmax','delta_m','mu_g_low','sigma_g_low','lambda_g_low','mu_g_high','sigma_g_high','lambda_g',
                        'mmin_NS','mmax_NS','delta_m_NS']
    def update(self,**kwargs):
        p1=SmoothedProb(PowerLawTwoGaussians(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],
                                             kwargs['lambda_g'],kwargs['lambda_g_low'],kwargs['mu_g_low'],
                                             kwargs['sigma_g_low'],kwargs['mmin'],kwargs['mu_g_low']+5*kwargs['sigma_g_low'],
                                             kwargs['mu_g_high'],kwargs['sigma_g_high'],kwargs['mmin'],kwargs['mu_g_high']+5*kwargs['sigma_g_high']),kwargs['delta_m'])
        p2=SmoothedProb(PowerLaw(kwargs['mmin_NS'],kwargs['mmax_NS'],kwargs['beta']),kwargs['delta_m_NS'])
        self.prior=conditional_2dimpdf(p1,p2)
        
class spinprior_default(object):
    def __init__(self):
        self.population_parameters=['alpha_chi','beta_chi','sigma_t','csi_spin']
        self.event_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
        self.name='DEFAULT'

    def update(self,**kwargs):
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        self.csi_spin = kwargs['csi_spin']
        self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
        self.beta_pdf = BetaDistribution(self.alpha_chi,self.beta_chi)
    def log_pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        return self.beta_pdf.log_pdf(chi_1)+self.beta_pdf.log_pdf(chi_2)+xp.log(self.csi_spin*self.aligned_pdf.pdf(cos_t_1)+(1.-self.csi_spin)*0.5)+xp.log(self.csi_spin*self.aligned_pdf.pdf(cos_t_2)+(1.-self.csi_spin)*0.5)
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2))
    
class spinprior_gaussian(object):
    def __init__(self):
        self.population_parameters=['mu_chi_eff','sigma_chi_eff','mu_chi_p','sigma_chi_p','rho']
        self.event_parameters=['chi_eff','chi_p']
        self.name='GAUSSIAN'
    def update(self,**kwargs):
        self.pdf_evaluator=Bivariate2DGaussian(x1min=-1.,x1max=1.,x1mean=kwargs['mu_chi_eff'],
                                               x2min=0.,x2max=1.,x2mean=kwargs['mu_chi_p'],
                                               x1variance=kwargs['sigma_chi_eff']**2.,x12covariance=kwargs['rho']*kwargs['sigma_chi_eff']*kwargs['sigma_chi_p'],
                                               x2variance=kwargs['sigma_chi_p']**2.)
    def log_pdf(self,chi_eff,chi_p):
        return self.pdf_evaluator.log_pdf(chi_eff,chi_p)
    def pdf(self,chi_eff,chi_p):
        return xp.exp(self.log_pdf(chi_eff,chi_p))
        
        
        
        
class spinprior_ECOs(object):
    def __init__(self):
        self.population_parameters=['alpha_chi','beta_chi','eps', 'R', 'f_eco', 'sigma']
        self.event_parameters=['chi_1','chi_2'] 
        self.name='DEFAULT'
        
    def get_chi_crit(self, eps, R):
        return 0.5

    def update(self,**kwargs):
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        self.eps = kwargs['eps']
        self.R = kwargs['R']
        self.f_eco = kwargs['f_eco']
        self.sigma = kwargs['sigma']
        self.chi_crit = self.get_chi_crit(self.eps,self.R)
        #self.aligned_pdf = TruncatedGaussian(1.,kwargs['sigma_t'],-1.,1.)
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
            
        self.beta_pdf = BetaDistribution(self.alpha_chi,self.beta_chi)
        self.truncatedbeta_pdf = TruncatedBetaDistribution(self.alpha_chi,self.beta_chi,self.chi_crit)
        self.truncatedgaussian_pdf = TruncatedGaussian(self.chi_crit, self.sigma, 0., 1.)
        self.lambda_eco = 1-self.beta_pdf.cdf(xp.array([self.get_chi_crit(self.eps, self.R)]))[0]
        
        
    def pdf(self,chi_1,chi_2):
        p_chi_1 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_1) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_1)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_1) 
        p_chi_2 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_2) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_2)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_2) 
        return p_chi_1*p_chi_2
        
        
    def log_pdf(self,chi_1,chi_2):
        return xp.log(self.pdf(chi_1,chi_2))
    
################ END: Small wrappers used in support of the main wrappers above ###################

        
            
        
