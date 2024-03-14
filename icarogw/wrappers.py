from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn
from .cosmology import alphalog_astropycosmology, cM_astropycosmology, extraD_astropycosmology, Xi0_astropycosmology, astropycosmology
from .cosmology import  md_rate, powerlaw_rate
from .priors import SmoothedProb, PowerLaw, BetaDistribution, TruncatedBetaDistribution, TruncatedGaussian, Bivariate2DGaussian
from .priors import PowerLawGaussian, BrokenPowerLaw, PowerLawTwoGaussians, absL_PL_inM, conditional_2dimpdf, piecewise_constant_2d_distribution_normalized
import copy
from astropy.cosmology import FlatLambdaCDM, FlatwCDM

# LVK Reviewed
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

# LVK Reviewed
def rate_wrappers_init(name):
    if name == 'PowerLaw':
        rate_wrap=rateevolution_PowerLaw()
    elif name == 'Madau':
        rate_wrap=rateevolution_Madau()
    else:
        raise ValueError('Rate model not known')    
    return rate_wrap

# LVK Reviewed
def cosmology_wrappers_init(name,zmax):
    if name == 'FlatLambdaCDM':
        cosmology_wrap=FlatLambdaCDM_wrap(zmax=zmax)
    elif name == 'FlatwCDM':
        cosmology_wrap=FlatwCDM_wrap(zmax=zmax)
    else:
        raise ValueError('Cosmology model not known')
    return cosmology_wrap

# LVK Reviewed
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
# LVK Reviewed
class source_mass_default(object):
    def pdf(self,mass_1_source,mass_2_source):
        return self.prior.pdf(mass_1_source,mass_2_source)
    def log_pdf(self,mass_1_source,mass_2_source):
        return self.prior.log_pdf(mass_1_source,mass_2_source)

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

# LVK Reviewed
class massprior_PowerLaw(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha','beta','mmin','mmax']
    def update(self,**kwargs):
        p1,p2=PowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha']),PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta'])
        self.prior=conditional_2dimpdf(p1,p2)

class massprior_pairedQ_PowerLawPeak(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha','beta','mmin','mmax','delta_m','mu_g','sigma_g','lambda_peak']
    def update(self,**kwargs):
        p=SmoothedProb(PowerLawGaussian(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],kwargs['lambda_peak'],kwargs['mu_g'],
                                         kwargs['sigma_g'],kwargs['mmin'],kwargs['mu_g']+5*kwargs['sigma_g']),kwargs['delta_m'])

        def pairing_function(m1,m2,beta=kwargs['beta']):
            q = m2/m1
            toret = xp.power(q,beta)
            toret[q>1] = 0.
            return toret
        
        self.prior=paired_2dimpdf(p,pairing_function)

# LVK Reviewed
class massprior_PowerLawPeak(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha','beta','mmin','mmax','delta_m','mu_g','sigma_g','lambda_peak']
    def update(self,**kwargs):
        p1=SmoothedProb(PowerLawGaussian(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],kwargs['lambda_peak'],kwargs['mu_g'],
                                         kwargs['sigma_g'],kwargs['mmin'],kwargs['mu_g']+5*kwargs['sigma_g']),kwargs['delta_m'])
        p2=SmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimpdf(p1,p2)

# LVK Reviewed
class massprior_BrokenPowerLaw(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','beta','mmin','mmax','delta_m','b']
    def update(self,**kwargs):
        p1=SmoothedProb(BrokenPowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b']),kwargs['delta_m'])
        p2=SmoothedProb(PowerLaw(kwargs['mmin'],kwargs['mmax'],kwargs['beta']),kwargs['delta_m'])
        self.prior=conditional_2dimpdf(p1,p2)

# LVK Reviewed
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

# LVK Reviewed
class massprior_PowerLaw_NSBH(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha','beta','mmin','mmax','mmin_NS','mmax_NS']
    def update(self,**kwargs):
        p1,p2=PowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha']),PowerLaw(kwargs['mmin_NS'],kwargs['mmax_NS'],kwargs['beta'])
        self.prior=conditional_2dimpdf(p1,p2)

# LVK Reviewed
class massprior_PowerLawPeak_NSBH(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha','beta','mmin','mmax','delta_m','mu_g','sigma_g','lambda_peak','mmin_NS','mmax_NS','delta_m_NS']
    def update(self,**kwargs):
        p1=SmoothedProb(PowerLawGaussian(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha'],kwargs['lambda_peak'],kwargs['mu_g'],
                                         kwargs['sigma_g'],kwargs['mmin'],kwargs['mu_g']+5*kwargs['sigma_g']),kwargs['delta_m'])
        p2=SmoothedProb(PowerLaw(kwargs['mmin_NS'],kwargs['mmax_NS'],kwargs['beta']),kwargs['delta_m_NS'])
        self.prior=conditional_2dimpdf(p1,p2)

# LVK Reviewed
class massprior_BrokenPowerLaw_NSBH(source_mass_default):
    def __init__(self):
        self.population_parameters=['alpha_1','alpha_2','beta','mmin','mmax','delta_m','b','mmin_NS','mmax_NS','delta_m_NS']
    def update(self,**kwargs):
        p1=SmoothedProb(BrokenPowerLaw(kwargs['mmin'],kwargs['mmax'],-kwargs['alpha_1'],-kwargs['alpha_2'],kwargs['b']),kwargs['delta_m'])
        p2=SmoothedProb(PowerLaw(kwargs['mmin_NS'],kwargs['mmax_NS'],kwargs['beta']),kwargs['delta_m_NS'])
        self.prior=conditional_2dimpdf(p1,p2)

# LVK Reviewed
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

class massprior_BinModel2d(source_mass_default):
    def __init__(self, n_bins_1d):
        self.population_parameters=['mmin','mmax']
        n_bins_total = int(n_bins_1d * (n_bins_1d + 1) / 2)
        self.bin_parameter_list = ['bin_' + str(i) for i in range(n_bins_total)]
        self.population_parameters += self.bin_parameter_list
    def update(self,**kwargs):
        kwargs_bin_parameters = xp.array([kwargs[key] for key in self.bin_parameter_list])
        
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
      
class spinprior_ECOs_totally_reflective(object):
    def __init__(self,q=1.):
        # q=1 is the polar case, q = 2 is the axial case, m=2 fixed
        self.q=q
        self.population_parameters=['alpha_chi','beta_chi','eps', 'f_eco', 'sigma_chi_ECO']
        self.event_parameters=['chi_1','chi_2'] 
        self.name='DEFAULT'
        
    def get_chi_crit(self, eps):
        xp = get_module_array(eps)   
        return xp.pi*(1.+self.q)/(2*xp.abs(xp.log(eps)))

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
        self.truncatedgaussian_pdf = TruncatedGaussian(self.chi_crit, self.sigma, 0., self.chi_crit)
        self.lambda_eco = 1-self.beta_pdf.cdf(np.array([self.get_chi_crit(self.eps)]))[0]
        
        
    def pdf(self,chi_1,chi_2):
        p_chi_1 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_1) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_1)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_1) 
        p_chi_2 = self.f_eco*((1-self.lambda_eco)*self.truncatedbeta_pdf.pdf(chi_2) + self.lambda_eco*self.truncatedgaussian_pdf.pdf(chi_2)) + (1-self.f_eco)*self.beta_pdf.pdf(chi_2) 
        return p_chi_1*p_chi_2
        
        
    def log_pdf(self,chi_1,chi_2):
        xp = get_module_array(chi_1)
        return xp.log(self.pdf(chi_1,chi_2))
    

        
            
        
