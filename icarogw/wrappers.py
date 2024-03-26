from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn
from .cosmology import alphalog_astropycosmology, cM_astropycosmology, extraD_astropycosmology, Xi0_astropycosmology, astropycosmology
from .cosmology import  md_rate, powerlaw_rate, beta_rate, beta_rate_line
from .priors import LowpassSmoothedProb, PowerLaw, BetaDistribution, TruncatedBetaDistribution, TruncatedGaussian, Bivariate2DGaussian, SmoothedPlusDipProb
from .priors import  EvolvingPowerLawPeak, PowerLawGaussian, BrokenPowerLaw, PowerLawTwoGaussians, absL_PL_inM, conditional_2dimpdf, conditional_2dimz_pdf, piecewise_constant_2d_distribution_normalized,paired_2dimpdf
from .priors import _lowpass_filter, _mixed_sigmoid_function
import copy
from astropy.cosmology import FlatLambdaCDM, FlatwCDM


class mixed_mass_redshift_evolving(object):

    def __init__(self,mw):
        self.population_parameters = mw.population_parameters + ['zt', 'delta_zt', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1']
        self.mw_red_ind = mw

    def update(self,**kwargs):
        self.mw_red_ind.update(**{key:kwargs[key] for key in self.mw_red_ind.population_parameters})
        self.zt = kwargs['zt']
        self.delta_zt = kwargs['delta_zt']
        self.mu_z0 = kwargs['mu_z0']
        self.mu_z1 = kwargs['mu_z1']
        self.sigma_z0 = kwargs['sigma_z0']
        self.sigma_z1 = kwargs['sigma_z1']

    def pdf(self,m,z):
        xp = get_module_array(m)
        wz = _lowpass_filter(z,self.zt,self.delta_zt)/_lowpass_filter(xp.array([0.]),self.zt,self.delta_zt)
        muz = self.mu_z0 + self.mu_z1*z
        sigmaz = self.sigma_z0 + self.sigma_z1*z
        gaussian_part = (xp.power(2*xp.pi,-0.5)/sigmaz) * xp.exp(-.5*xp.power((m-muz)/sigmaz,2.))
        return wz*self.mw_red_ind.pdf(m) + (1-wz)*gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))

class mixed_mass_redshift_evolving_model_trunc(object):

    def __init__(self,mw):
        self.population_parameters = mw.population_parameters + ['zt', 'delta_zt', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1']
        self.mw_red_ind = mw

    def update(self,**kwargs):
        self.mw_red_ind.update(**{key:kwargs[key] for key in self.mw_red_ind.population_parameters})
        self.zt = kwargs['zt']
        self.delta_zt = kwargs['delta_zt']
        self.mu_z0 = kwargs['mu_z0']
        self.mu_z1 = kwargs['mu_z1']
        self.sigma_z0 = kwargs['sigma_z0']
        self.sigma_z1 = kwargs['sigma_z1']

    def pdf(self,m,z):
        xp = get_module_array(m)
        wz = _lowpass_filter(z,self.zt,self.delta_zt)/_lowpass_filter(xp.array([0.]),self.zt,self.delta_zt)
        muz = self.mu_z0 + self.mu_z1*z
        sigmaz = self.sigma_z0 + self.sigma_z1*z
        gaussian_part = (xp.power(2*xp.pi,-0.5)/sigmaz) * xp.exp(-.5*xp.power((m-muz)/sigmaz,2.))

        if xp.any((muz - 3*sigmaz) < 0):    # Check that the gaussian peak excludes negative values for the masses at 3 sigma
            return xp.nan # Return nans as the log-likelihood put them at -inf
        else:
            return wz*self.mw_red_ind.pdf(m) + (1-wz)*gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


class mixed_mass_redshift_evolving_sigmoid(object):

    def __init__(self,mw):
        
        self.population_parameters = mw.population_parameters + ['zt', 'delta_zt', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1', 'mix_z0']
        self.mw_red_ind = mw

    def update(self,**kwargs):

        self.mw_red_ind.update(**{key:kwargs[key] for key in self.mw_red_ind.population_parameters})
        self.zt = kwargs['zt']
        self.delta_zt = kwargs['delta_zt']
        self.mu_z0 = kwargs['mu_z0']
        self.mu_z1 = kwargs['mu_z1']
        self.sigma_z0 = kwargs['sigma_z0']
        self.sigma_z1 = kwargs['sigma_z1']
        self.mix_z0 = kwargs['mix_z0']

    def pdf(self,m,z):

        xp = get_module_array(m)
        sx = get_module_array_scipy(m)
        wz = _mixed_sigmoid_function(z, self.zt, self.delta_zt, self.mix_z0)
        muz = self.mu_z0 +  self.mu_z1*z
        sigmaz = self.sigma_z0 + self.sigma_z1*z
        a, b = (0. - muz) / sigmaz, (xp.inf - muz) / sigmaz 
        gaussian_part = sx.stats.truncnorm.pdf(m,a,b,loc=muz,scale=sigmaz)
        return wz*self.mw_red_ind.pdf(m) + (1-wz)*gaussian_part
    
    def log_pdf(self,m,z):
        xp = get_module_array(m)
        return xp.log(self.pdf(m,z))


# A parent class for the rate
# LVK Reviewed
class rate_default(object):
    def evaluate(self,z):
        return self.rate.evaluate(z)
    def log_evaluate(self,z):
        return self.rate.log_evaluate(z)
    
# LVK Reviewed
class rateevolution_PowerLaw(rate_default):
    def __init__(self):
        self.population_parameters=['gamma']
    def update(self,**kwargs):
        self.rate=powerlaw_rate(**kwargs)

# LVK Reviewed
class rateevolution_Madau(rate_default):
    def __init__(self):
        self.population_parameters=['gamma','kappa','zp']
    def update(self,**kwargs):
        self.rate=md_rate(**kwargs)

class rateevolution_beta(rate_default):
    def __init__(self):
        self.population_parameters=['a','b','c']
    def update(self,**kwargs):
        self.rate=beta_rate(**kwargs)

class rateevolution_beta_line(rate_default):
    def __init__(self):
        self.population_parameters=['a','b','c','d']
    def update(self,**kwargs):
        self.rate=beta_rate_line(**kwargs)

# LVK Reviewed
class FlatLambdaCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=FlatLambdaCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))

# LVK Reviewed
class FlatwCDM_wrap(object):
    def __init__(self,zmax):
        self.population_parameters=['H0','Om0','w0']
        self.cosmology=astropycosmology(zmax)
        self.astropycosmo=FlatwCDM
    def update(self,**kwargs):
        self.cosmology.build_cosmology(self.astropycosmo(**kwargs))

# LVK Reviewed
class Xi0_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['Xi0','n']
        self.cosmology=Xi0_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),Xi0=kwargs['Xi0'],n=kwargs['n'])

# LVK Reviewed
class extraD_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['D','n','Rc']
        self.cosmology=extraD_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),D=kwargs['D'],n=kwargs['n'],Rc=kwargs['Rc'])

# LVK Reviewed
class cM_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['cM']
        self.cosmology=cM_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),cM=kwargs['cM'])

# LVK Reviewed
class alphalog_mod_wrap(object):
    def __init__(self,bgwrap):
        self.bgwrap=copy.deepcopy(bgwrap)
        self.population_parameters=self.bgwrap.population_parameters+['alphalog_1','alphalog_2','alphalog_3']
        self.cosmology=alphalog_astropycosmology(bgwrap.cosmology.zmax)
    def update(self,**kwargs):
        bgdict={key:kwargs[key] for key in self.bgwrap.population_parameters}
        self.cosmology.build_cosmology(self.bgwrap.astropycosmo(**bgdict),alphalog_1=kwargs['alphalog_1']
                                       ,alphalog_2=kwargs['alphalog_2'],alphalog_3=kwargs['alphalog_3'])

# A parent class for the standard 1 D mass probabilities
class pm_prob(object):
    def pdf(self,mass_1_source):
        return self.prior.pdf(mass_1_source)
    def log_pdf(self,mass_1_source):
        return self.prior.log_pdf(mass_1_source)

class mass_ratio_prior_Gaussian(pm_prob):
    def __init__(self):
        self.population_parameters=['mu_q','sigma_q']
    def update(self,**kwargs):
        p1=TruncatedGaussian(kwargs['mu_q'],kwargs['sigma_q'],0.,1.)
        self.prior=p1

class lowSmoothedwrapper(object):
   def __init__(self, mw):
        self.population_parameters = ['delta_m'] + mw.population_parameters
        self.mw = mw
   def update(self,**kwargs):
        self.mw.update(**{key:kwargs[key] for key in self.mw.population_parameters})
        self.p1 = LowpassSmoothedProb(self.mw.prior,kwargs['delta_m'])
   def pdf(self,m):
        return self.p1.pdf(m)
   def log_pdf(self,m):
        return self.p1.log_pdf(m)

# A parent class for the standard mass probabilities
# LVK Reviewed
class pm1m2_prob(object):
    def pdf(self,mass_1_source,mass_2_source):
        return self.prior.pdf(mass_1_source,mass_2_source)
    def log_pdf(self,mass_1_source,mass_2_source):
        return self.prior.log_pdf(mass_1_source,mass_2_source)
    
class pm1m2z_prob(object):
    def pdf(self,mass_1_source,mass_2_source,z):
        return self.prior.pdf(mass_1_source,mass_2_source,z)
    def log_pdf(self,mass_1_source,mass_2_source,z):
        return self.prior.log_pdf(mass_1_source,mass_2_source,z)

class massprior_PowerLaw(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax']
    def update(self,**kwargs):
        self.prior=PowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'])
        
class massprior_PowerLawPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax','mu_g','sigma_g','lambda_peak']
    def update(self,**kwargs):
        self.prior=PowerLawGaussian(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],kwargs['lambda_peak'],kwargs['mu_g'],
                                         kwargs['sigma_g'],kwargs['mmin'],kwargs['mu_g']+5*kwargs['sigma_g'])
        
class massprior_BrokenPowerLaw(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','mmin','mmax','b']
    def update(self,**kwargs):
        self.prior=BrokenPowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b'])
        
class massprior_MultiPeak(pm_prob):
    def __init__(self):
        self.population_parameters=['alpha','mmin','mmax','mu_g_low','sigma_g_low','lambda_g_low','mu_g_high','sigma_g_high','lambda_g']
    def update(self,**kwargs):
        self.prior=PowerLawTwoGaussians(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],
                                             kwargs['lambda_g'],kwargs['lambda_g_low'],kwargs['mu_g_low'],
                                             kwargs['sigma_g_low'],kwargs['mmin'],kwargs['mu_g_low']+5*kwargs['sigma_g_low'],
                                             kwargs['mu_g_high'],kwargs['sigma_g_high'],kwargs['mmin'],kwargs['mu_g_high']+5*kwargs['sigma_g_high'])
        
class massprior_EvolvingPowerLawPeak(object):
    def __init__(self,mw):
        self.population_parameters = mw.population_parameters + ['zt', 'delta_zt', 'mu_z0', 'mu_z1', 'sigma_z0', 'sigma_z1']
        self.mw_nonevolving = mw
    def update(self,**kwargs):
        self.mw_nonevolving.update(**{key:kwargs[key] for key in self.mw_nonevolving.population_parameters})
        self.zt = kwargs['zt']
        self.delta_zt = kwargs['delta_zt']
        self.mu_z0 = kwargs['mu_z0']
        self.mu_z1 = kwargs['mu_z1']
        self.sigma_z0 = kwargs['sigma_z0']
        self.sigma_z1 = kwargs['sigma_z1']
        self.prior = EvolvingPowerLawPeak(self.mw_nonevolving, self.zt, self.delta_zt, self.mu_z0, self.mu_z1, self.sigma_z0, self.sigma_z1)

class m1m2_conditioned(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = self.wrapper_m.prior
        p2 = PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta'])
        self.prior=conditional_2dimpdf(p1,p2)

class m1m2_conditioned_lowpass_m2(pm1m2z_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = self.wrapper_m.prior
        p2 = LowpassSmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimz_pdf(p1,p2)

class m1m2_conditioned_lowpass(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters+['beta','delta_m']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p1 = LowpassSmoothedProb(self.wrapper_m.prior,kwargs['delta_m'])
        p2 = LowpassSmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimpdf(p1,p2)

class m1m2_paired_massratio_dip(pm1m2_prob):
    def __init__(self,wrapper_m):
        self.population_parameters = wrapper_m.population_parameters + ['beta','bottomsmooth', 'topsmooth', 
                                                                        'leftdip','rightdip','leftdipsmooth', 
                                                                        'rightdipsmooth','deep']
        self.wrapper_m = wrapper_m
    def update(self,**kwargs):
        self.wrapper_m.update(**{key:kwargs[key] for key in self.wrapper_m.population_parameters})
        p = SmoothedPlusDipProb(self.wrapper_m.prior,**{key:kwargs[key] for key in ['bottomsmooth', 'topsmooth', 
                                                                        'leftdip', 'rightdip', 
                                                                        'leftdipsmooth','rightdipsmooth','deep']})
        def pairing_function(m1,m2,beta=kwargs['beta']):
            xp = get_module_array(m1)
            q = m2/m1
            toret = xp.power(q,beta)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)

class massprior_BinModel2d(pm1m2_prob):
    def __init__(self, n_bins_1d):
        self.population_parameters=['mmin','mmax']
        n_bins_total = int(n_bins_1d * (n_bins_1d + 1) / 2)
        self.bin_parameter_list = ['bin_' + str(i) for i in range(n_bins_total)]
        self.population_parameters += self.bin_parameter_list
    def update(self,**kwargs):
        kwargs_bin_parameters = np.array([kwargs[key] for key in self.bin_parameter_list])
        
        pdf_dist = piecewise_constant_2d_distribution_normalized(
            kwargs['mmin'], 
            kwargs['mmax'],
            kwargs_bin_parameters
        )
        
        self.prior=pdf_dist
        
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
        xp = get_module_array(chi_1)
        return self.beta_pdf.log_pdf(chi_1)+self.beta_pdf.log_pdf(chi_2)+xp.log(self.csi_spin*self.aligned_pdf.pdf(cos_t_1)+(1.-self.csi_spin)*0.5)+xp.log(self.csi_spin*self.aligned_pdf.pdf(cos_t_2)+(1.-self.csi_spin)*0.5)
    def pdf(self,chi_1,chi_2,cos_t_1,cos_t_2):
        xp = get_module_array(chi_1)
        return xp.exp(self.log_pdf(chi_1,chi_2,cos_t_1,cos_t_2))

# LVK Reviewed
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
        xp = get_module_array(chi_eff)
        return xp.exp(self.log_pdf(chi_eff,chi_p))
      
class spinprior_ECOs(object):
    def __init__(self):
        self.population_parameters=['alpha_chi','beta_chi','eps', 'f_eco', 'sigma_chi_ECO']
        self.event_parameters=['chi_1','chi_2'] 
        self.name='DEFAULT'
        
    def get_chi_crit(self, eps):
        xp = get_module_array(eps)
        q = 1. # Value for polar perturbations, more conservative
        return xp.pi*(1.+q)/(2*xp.abs(xp.log10(eps)))

    def update(self,**kwargs):
        self.alpha_chi = kwargs['alpha_chi']
        self.beta_chi = kwargs['beta_chi']
        self.eps = kwargs['eps']
        self.f_eco = kwargs['f_eco']
        self.sigma = kwargs['sigma_chi_ECO']
        self.chi_crit = self.get_chi_crit(self.eps)
        if (self.alpha_chi <= 1) | (self.beta_chi <= 1) :
            raise ValueError('Alpha and Beta must be > 1') 
            
        self.beta_pdf = BetaDistribution(self.alpha_chi,self.beta_chi)
        self.truncatedbeta_pdf = TruncatedBetaDistribution(self.alpha_chi,self.beta_chi,self.chi_crit)
        self.truncatedgaussian_pdf = TruncatedGaussian(self.chi_crit, self.sigma, 0., 1.)
        self.lambda_eco = 1-self.beta_pdf.cdf(np.array([self.get_chi_crit(self.eps)]))[0]
        
        
    def pdf(self,chi_1,chi_2):
        p_chi_1 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_1) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_1)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_1) 
        p_chi_2 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_2) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_2)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_2) 
        return p_chi_1*p_chi_2
        
        
    def log_pdf(self,chi_1,chi_2):
        xp = get_module_array(chi_1)
        return xp.log(self.pdf(chi_1,chi_2))
    

        
            
        
