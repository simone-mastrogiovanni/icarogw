from .cupy_pal import *
from .cosmology import *
from .conversions import detector2source_jacobian, detector2source
from .priors import *
from scipy.stats import gaussian_kde
import copy

from astropy.cosmology import FlatLambdaCDM, FlatwCDM



################ Big wrappers below

class CBC_vanilla_EM_counterpart(object):
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
            
        self.GW_parameters = ['mass_1', 'mass_2', 'luminosity_distance','z_EM']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            self.GW_parameters = self.GW_parameters + self.sw.GW_parameters
            
    def update(self,**kwargs):
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        
        if len(kwargs['mass_1'].shape) != 2:
            raise ValueError('The EM counterpart rate wants N_ev x N_samples arrays')
        
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.GW_parameters})
        
        n_ev = kwargs['mass_1'].shape[0]
        log_weights = xp.empty(kwargs['z_EM'].shape)
        for i in range(n_ev): 
            ww = xp.exp(log_weights[i,:])
            kde_fit = gaussian_kde(z[i,:],weights=ww)
            log_weights[i,:] = np.log(xp.sum(ww)/kwargs['mass_1'].shape[1])+kde_fit.logpdf(kwargs['z_EM'][i,:])

        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.GW_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out


class CBC_vanilla_rate(object):
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
            
        self.GW_parameters = ['mass_1', 'mass_2', 'luminosity_distance']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            self.GW_parameters = self.GW_parameters + self.sw.GW_parameters
            
    def update(self,**kwargs):
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            self.R0 = kwargs['R0']
        
    def log_rate_PE(self,prior,**kwargs):
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.GW_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology) 
        log_dVc_dz=xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z)*4*xp.pi)
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+log_dVc_dz \
        -xp.log(prior)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log1p(z)
        
        if self.sw is not None:
            log_weights+=self.sw.log_pdf(**{key:kwargs[key] for key in self.sw.GW_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.R0)
        else:
            log_out = log_weights
            
        return log_out
    
class CBC_catalog_vanilla_rate(object):
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
            
        self.GW_parameters = ['mass_1', 'mass_2', 'luminosity_distance','sky_indices']
        
        if self.sw is not None:
            self.population_parameters = self.population_parameters+self.sw.population_parameters
            self.GW_parameters = self.GW_parameters + self.sw.GW_parameters
            
    def update(self,**kwargs):
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.mw.update(**{key: kwargs[key] for key in self.mw.population_parameters})
        self.rw.update(**{key: kwargs[key] for key in self.rw.population_parameters})
        if self.sw is not None:
            self.sw.update(**{key: kwargs[key] for key in self.sw.population_parameters})
            
        if not self.scale_free:
            
            self.Rgal = kwargs['Rgal']
        
    def log_rate_PE(self,prior,**kwargs):
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology)
        dNgal_cat,dNgal_bg=self.catalog.effective_galaxy_number_interpolant(z,kwargs['sky_indices'],self.cw.cosmology
                                                    ,dl=kwargs['luminosity_distance'],average=False)

        # Effective number density of galaxies (Eq. 2.19 on the overleaf document)
        dNgaleff=dNgal_cat+dNgal_bg
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+xp.log(dNgaleff) \
        -xp.log1p(z)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log(prior)
        
        if self.sw is not None:
            log_weights+=self.spin_wrap.log_pdf(**{key:self.posterior_parallel[key] for key in self.sw.GW_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.Rgal)
        else:
            log_out = log_weights
            
        return log_out
    
    def log_rate_injections(self,prior,**kwargs):
        ms1, ms2, z = detector2source(kwargs['mass_1'],kwargs['mass_2'],kwargs['luminosity_distance'],self.cw.cosmology)
        dNgal_cat,dNgal_bg=self.catalog.effective_galaxy_number_interpolant(z,kwargs['sky_indices'],self.cw.cosmology
                                                    ,dl=kwargs['luminosity_distance'],average=self.average)

        # Effective number density of galaxies (Eq. 2.19 on the overleaf document)
        dNgaleff=dNgal_cat+dNgal_bg
        
        # Sum over posterior samples in Eq. 1.1 on the icarogw2.0 document
        log_weights=self.mw.log_pdf(ms1,ms2)+self.rw.rate.log_evaluate(z)+xp.log(dNgaleff) \
        -xp.log1p(z)-xp.log(detector2source_jacobian(z,self.cw.cosmology))-xp.log(prior)
        
        if self.sw is not None:
            log_weights+=self.spin_wrap.log_pdf(**{key:self.posterior_parallel[key] for key in self.sw.GW_parameters})
            
        if not self.scale_free:
            log_out = log_weights + xp.log(self.Rgal)
        else:
            log_out = log_weights
            
        return log_out

################ Small wrappers below

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
        self.GW_parameters=['chi_1','chi_2','cos_t_1','cos_t_2']
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
        self.GW_parameters=['chi_eff','chi_p']
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
    
        
        
        


        
            
        
