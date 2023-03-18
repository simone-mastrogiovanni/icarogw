from .cupy_pal import *
from scipy.integrate import cumtrapz
import mpmath

COST_C= 299792.458 # Speed of light in km/s

class base_cosmology(object):
    def __init__(self,zmax):
        '''
        A class to handle cosmology. Note, quantities related to comoving volume are in Gpc while quantities related to distances 
        in Mpc.
        
        Parameters
        ----------
        zmax: float
            initialize the cosmology up to zmax
        '''
        self.zmax=zmax
        self.z_cpu=np.logspace(-6,np.log10(self.zmax),2500)
        self.z_gpu=xp.logspace(-6,np.log10(self.zmax),2500)
        
        self.log10_z_gpu=xp.log10(self.z_gpu)
        
    def z2dl(self,z):
        '''
        Converts redshift to luminosity distance
        
        Parameters
        ----------
        z: xp.array
            Redshift
        
        Reutrns
        -------
        dl: xp.array
            luminosity distance in Mpc
        ''' 
        origin=z.shape
        ravelled=xp.ravel(xp.log10(z))
        interpo=xp.interp(ravelled,self.log10_z_gpu,self.log10_dl_at_z)
        return xp.reshape(10**interpo,origin)
    
    def z2Vc(self,z):
        '''
        Converts redshift to luminosity distance
        
        Parameters
        ----------
        z: xp.array
            Redshift
        
        Reutrns
        -------
        dl: xp.array
            luminosity distance in Mpc
        ''' 
        origin=z.shape
        ravelled=xp.ravel(xp.log10(z))
        interpo=xp.interp(ravelled,self.log10_z_gpu,self.log10_Vc)
        return xp.reshape(10**interpo,origin)
        
        
    def dl2z(self,dl):
        '''
        Converts luminosity distance to redshift
        
        Parameters
        ----------
        dl: xp.array
            luminosity distance in Mpc
        
        Reutrns
        -------
        z: xp.array
            redshift
        '''
        origin=dl.shape
        ravelled=xp.ravel(xp.log10(dl))
        interpo=xp.interp(ravelled,self.log10_dl_at_z,self.log10_z_gpu)
        return xp.reshape(10**interpo,origin)
    
    def dVc_by_dzdOmega_at_z(self,z):
        '''
        Calculates the differential of the comoving volume per sterdian at a given redshift
        
        Parameters
        ----------
        z: xp.array
            Redshift
        
        Reutrns
        -------
        dVc_by_dzdOmega: xp.array
            comoving volume per sterdian at a given redshift in Gpc3std-1
        '''
        origin=z.shape
        ravelled=xp.ravel(xp.log10(z))
        interpo=xp.interp(ravelled,self.log10_z_gpu,self.log10_dVc_dzdOmega)
        return xp.reshape(10**interpo,origin)
    
    def ddl_by_dz_at_z(self,z):
        '''
        Calculates the differential of the luminosity distance at given redshift
        
        Parameters
        ----------
        z: xp.array
            Redshift
        
        Reutrns
        -------
        ddl_by_dz: xp.array
            differential of the luminosity distance in Mpc
        '''
        origin=z.shape
        ravelled=xp.ravel(xp.log10(z))
        interpo=xp.interp(ravelled,self.log10_z_gpu,self.log10_ddl_by_dz)
        return xp.reshape(10**interpo,origin)
    
    def sample_comoving_volume(self,Nsamp,zmin,zmax):
        '''
        Samples uniform in the comoving volume
        
        Parameters
        ----------
        Nsamp: int
            Number of samples to draw
        zmin, zmax: float
            Minimum and maxmium redshift at which to draw
        
        Reutrns
        -------
        z_samples: xp.array
            Distribution in z uniform in comoving volume
        '''
        zproxy=xp.linspace(zmin,zmax,10000)
        prob=self.dVc_by_dzdOmega_at_z(zproxy)
        cdf=xp.cumsum(prob)/prob.sum()
        cdf[0]=0.
        cdf_samps=xp.random.rand(Nsamp)
        return xp.interp(cdf_samps,cdf,zproxy)
    

class astropycosmology(base_cosmology):
    def  build_cosmology(self,astropy_cosmo):
        '''
        Construct the cosmology
        
        Parameters
        ----------
        astropy_cosmo: Astropy.cosmology class
            initialize the cosmology up to zmax
        '''
        self.astropy_cosmo=astropy_cosmo
        self.little_h=astropy_cosmo.H(0.).value/100.
        self.log10_dVc_dzdOmega=xp.log10(np2cp(astropy_cosmo.differential_comoving_volume(self.z_cpu).value))-9. # Conversion from Mpc to Gpc
        self.log10_Vc=xp.log10(np2cp(astropy_cosmo.comoving_volume(self.z_cpu).value))-9. # Conversion to Gpc
        self.log10_dl_at_z=xp.log10(np2cp(astropy_cosmo.luminosity_distance(self.z_cpu).value))
        self.log10_ddl_by_dz=xp.log10((xp.power(10.,self.log10_dl_at_z)/(1.+self.z_gpu))+COST_C*(1.+self.z_gpu)/np2cp(astropy_cosmo.H(self.z_cpu).value))
        
        
class extraD_astropycosmology(astropycosmology):
    def build_cosmology(self,astropy_cosmo,D,n,Rc):
        '''
        Construct the cosmology
        
        Parameters
        ----------
        astropy_cosmo: Astropy.cosmology class
            initialize the cosmology up to zmax
        D,n,Rc: float
            The GR modification parameters as described in Eq. 2.22 of https://arxiv.org/pdf/2109.08748.pdf
        '''
        
        super().build_cosmology(astropy_cosmo)
        dlem = xp.power(10.,self.log10_dl_at_z)
        dlbydz_em = xp.power(10.,self.log10_ddl_by_dz)
        self.log10_dl_at_z=xp.log10(dlem*xp.power(1+xp.power(dlem/((1.+self.z_gpu)*Rc),n),(D-4.)/(2*n)))
        Afa=1.+xp.power(dlem/((1.+self.z_gpu)*Rc),n)
        expo=(D-4.)/(2.*n)
        # We put the absolute value for the Jacobian (because this is needed for probabilities)
        self.log10_ddl_by_dz=xp.log10(xp.abs(xp.power(Afa,expo)*(dlbydz_em+xp.power(dlem/Rc,n)*(expo*n/Afa)*\
            (dlbydz_em/xp.power(1.+self.z_gpu,n)-dlem/xp.power(1.+self.z_gpu,n+1.)))))

class cM_astropycosmology(astropycosmology):
    def build_cosmology(self,astropy_cosmo,cM):
        '''
        Construct the cosmology
        
        Parameters
        ----------
        astropy_cosmo: Astropy.cosmology class
            initialize the cosmology up to zmax
        D,n,Rc: float
            The GR modification parameters as described in Eq. 2.22 of https://arxiv.org/pdf/2109.08748.pdf
        '''
        
        super().build_cosmology(astropy_cosmo)
        dlem = xp.power(10.,self.log10_dl_at_z)
        dlbydz_em = xp.power(10.,self.log10_ddl_by_dz)
        # Implementation of the running Planck mass model general case as described in the overleaf https://www.overleaf.com/project/62330c2859bb3c2a5982c2b6
        # Define array for numerical integration
        ZforI=np.append(self.z_cpu,self.z_cpu[-1]-self.z_cpu[-2])
        ZforI=np.append(0.,ZforI)
        Zhalfbin=(ZforI[:-1:]+ZforI[1::])*0.5
        Integrandhalfin=1./((1+Zhalfbin)*np.power(astropy_cosmo.efunc(Zhalfbin),2.))
        Integrand=np2cp(1./((1+self.z_cpu)*np.power(astropy_cosmo.efunc(self.z_cpu),2.)))
        Integral=np2cp(cumtrapz(Integrandhalfin,Zhalfbin))
        self.log10_dl_at_z=xp.log10(dlem*xp.exp((0.5*cM)*Integral))
        exp_factor = xp.exp(0.5*cM*Integral)
        # We put the absolute value for the Jacobian (because this is needed for probabilities)
        self.log10_ddl_by_dz=xp.log10(xp.abs(exp_factor*dlbydz_em+0.5*dlem*cM*exp_factor*Integrand))
        
        
    
        
        
class Xi0_astropycosmology(astropycosmology):
    def  build_cosmology(self,astropy_cosmo,Xi0,n):
        '''
        Construct the cosmology
        
        Parameters
        ----------
        astropy_cosmo: Astropy.cosmology class
            initialize the cosmology up to zmax
        Xi0, n: float
            The GR modification parameters as described in Eq. 2.31 of https://arxiv.org/pdf/1906.01593
        '''
        
        super().build_cosmology(astropy_cosmo)
        dlem = xp.power(10.,self.log10_dl_at_z)
        dlbydz_em = xp.power(10.,self.log10_ddl_by_dz)
        self.log10_dl_at_z=xp.log10(dlem*(Xi0+(1.-Xi0)*xp.power(1.+self.z_gpu,-n)))
        # We put the absolute value for the Jacobian (because this is needed for probabilities)
        self.log10_ddl_by_dz=xp.log10(xp.abs(dlbydz_em*(Xi0+(1.-Xi0)*xp.power(1+self.z_gpu,-n))-dlem*(1.-Xi0)*n*xp.power(1+self.z_gpu,-n-1)))
        
class alphalog_astropycosmology(astropycosmology):
    def  build_cosmology(self,astropy_cosmo,alphalog_1, alphalog_2, alphalog_3):
        '''
        Construct the cosmology
        
        Parameters
        ----------
        astropy_cosmo: Astropy.cosmology class
            initialize the cosmology up to zmax
        alphalog_1, alphalog_2, alphalog_3: float
            Implementation of the logarithm phenomenological dimension model proposed by the CosmologyTGR group
        '''
        super().build_cosmology(astropy_cosmo)
        dlem = xp.power(10.,self.log10_dl_at_z)
        dlbydz_em = xp.power(10.,self.log10_ddl_by_dz)
        self.log10_dl_at_z=xp.log10(dlem*(1.+alphalog_1*xp.log1p(self.z_gpu)+\
                                alphalog_2*(xp.log1p(self.z_gpu)**2.)+\
                                alphalog_3*(xp.log1p(self.z_gpu)**3.)))        
        part1 = dlbydz_em*(1.+alphalog_1*xp.log1p(self.z_gpu)+ \
                                        alphalog_2*(xp.log1p(self.z_gpu))**2.+\
                                        alphalog_3*(xp.log1p(self.z_gpu))**3.)
        part2 = (dlem/(1+self.z_gpu))*(alphalog_1+alphalog_2*2.*xp.log1p(self.z_gpu)+\
                      alphalog_3*3.*(xp.log1p(self.z_gpu)**2.))
        # We put the absolute value for the Jacobian (because this is needed for probabilities)
        self.log10_ddl_by_dz=xp.log10(xp.abs(part1 + part2))

    
class galaxy_MF(object):
    def __init__(self,band=None,Mmin=None,Mmax=None,Mstar=None,alpha=None,phistar=None):
        '''
        A class to handle the Schechter function in absolute magnitude
        
        Parameters
        ----------
        band: string
            W1, K or bJ band. Others are not implemented
        Mmin, Mmax,Mstar,alpha,phistar: float
            Minimum, maximum absolute magnitude. Knee-absolute magnitude (for h=1), Powerlaw factor and galaxy number density per Gpc-3 
        '''
        # Note, we convert phistar to Gpc-3
        if band is None:
            self.Mmin,self.Mmax,self.Mstar,self.alpha,self.phistar=Mmin,Mmax,Mstar,alpha,phistar
        else:
            if band=='W1':
                self.Mmin,self.Mmax,self.Mstar,self.alpha,self.phistar=-28, -16.6, -24.09, -1.12, 1.45e-2*1e9
            elif band=='bJ':
                self.Mmin,self.Mmax,self.Mstar,self.alpha,self.phistar=-22.00, -16.5, -19.66, -1.21, 1.61e-2*1e9
            elif band=='K':
                self.Mmin,self.Mmax,self.Mstar,self.alpha,self.phistar=-27.0, -19.0, -23.39, -1.09, 1.16e-2*1e9
            else:
                raise ValueError('Band not known')
    def build_MF(self,cosmology):
        '''
        Build the Magnitude function
        
        Parameters
        ----------
        cosmology: cosmology class
            cosmology class from the cosmology module
        '''
        self.cosmology=cosmology
        self.Mstarobs=self.Mstar+5*np.log10(cosmology.little_h)
        self.Mminobs=self.Mmin+5*np.log10(cosmology.little_h)
        self.Mmaxobs=self.Mmax+5*np.log10(cosmology.little_h)
        
        self.phistarobs=self.phistar*np.power(cosmology.little_h,3.)
        xmax=np.power(10.,0.4*(self.Mstarobs-self.Mminobs))
        xmin=np.power(10.,0.4*(self.Mstarobs-self.Mmaxobs))
        # Check if you need to replace this with a numerical integral.
        self.norm=self.phistarobs*float(mpmath.gammainc(self.alpha+1,a=xmin,b=xmax))

    def log_evaluate(self,M):
        '''
        Evluates the log of the Sch function
        
        Parameters
        ----------
        M: xp.array
            Absolute magnitude
            
        Returns
        -------
        log of the Sch function
        '''
        toret=xp.log(0.4*xp.log(10)*self.phistarobs)+ \
        ((self.alpha+1)*0.4*(self.Mstarobs-M))*xp.log(10.)-xp.power(10.,0.4*(self.Mstarobs-M))
        toret[(M<self.Mminobs) | (M>self.Mmaxobs)]=-xp.inf
        return toret

    def log_pdf(self,M):
        '''
        Evluates the log of the Sch function as pdf
        
        Parameters
        ----------
        M: xp.array
            Absolute magnitude
            
        Returns
        -------
        log of the Sch function as pdf
        '''
        return self.log_evaluate(M)-xp.log(self.norm)

    def pdf(self,M):
        '''
        Evluates the Sch as pdf
        
        Parameters
        ----------
        M: xp.array
            Absolute magnitude
            
        Returns
        -------
        log of the Sch as pdf
        '''
        return xp.exp(self.log_pdf(M))

    def evaluate(self,M):
        '''
        Evluates the Sch as pdf
        
        Parameters
        ----------
        M: xp.array
            Absolute magnitude
            
        Returns
        -------
        Sch function in Gpc-3
        '''
        return xp.exp(self.log_evaluate(M))

    def sample(self,N):
        '''
        Samples from the pdf
        
        Parameters
        ----------
        N: int
            Number of samples to generate
        
        Returns
        -------
        Samples: xp.array
        '''
        sarray=xp.linspace(self.Mminobs,self.Mmaxobs,10000)
        cdfeval=xp.cumsum(self.pdf(sarray))/self.pdf(sarray).sum()
        cdfeval[0]=0.
        randomcdf=xp.random.rand(N)
        return xp.interp(randomcdf,cdfeval,sarray,left=self.Mminobs,right=self.Mmaxobs)
    
    def build_effective_number_density_interpolant(self,epsilon):
        '''This method build the number density interpolant. This is defined as the integral from the Schecter function faint end to a value M_thr for the Schechter
        function times the luminosity weights. Note that this integral is done in x=Mstar-M, which is a cosmology independent quantity.
        
        Parameters
        ----------
        epsilon: float
            Powerlaw slope for the luminosity weights
        '''
        
        minv,maxv=self.Mmin,self.Mmax
        self.epsilon=epsilon
        Mvector_interpolant=np.linspace(minv,maxv,100)
        self.effective_density_interpolant=np.zeros_like(Mvector_interpolant)
        xmin=np.power(10.,0.4*(self.Mstar-maxv))
        for i in range(len(Mvector_interpolant)):
            xmax=np.power(10.,0.4*(self.Mstar-Mvector_interpolant[i]))
            self.effective_density_interpolant[i]=float(mpmath.gammainc(self.alpha+1+epsilon,a=xmin,b=xmax))
        
        self.effective_density_interpolant=np2cp(self.effective_density_interpolant)[::-1]
        self.xvector_interpolant=np2cp(self.Mstar-Mvector_interpolant)[::-1]
        
    def background_effective_galaxy_density(self,Mthr):
        '''Returns the effective galaxy density, i.e. dN_{gal,eff}/dVc, the effective number is given by the luminosity weights.
        
        Parameters
        ----------
        Mthr: xp.array
            Absolute magnitude threshold (faint) used to compute the integral
        '''
        
        origin=Mthr.shape
        ravelled=xp.ravel(self.Mstarobs-Mthr)
        outp=self.phistarobs*xp.interp(ravelled,self.xvector_interpolant,self.effective_density_interpolant
                           ,left=self.effective_density_interpolant[0],right=self.effective_density_interpolant[-1])
        return xp.reshape(outp,origin)
    
class kcorr(object):
    def __init__(self,band):
        '''
        A class to handle K-corrections
        
        Parameters
        ----------
        band: string
            W1, K or bJ band. Others are not implemented
        '''
        self.band=band
        if self.band not in ['W1','K','bJ']:
            raise ValueError('Band not known, please use W1 or K or bJ')
    def __call__(self,z):
        '''
        Evaluates the K-corrections at a given redshift, See Eq. 2 of https://arxiv.org/abs/astro-ph/0210394
        
        Parameters
        ----------
        z: xp.array
            Redshift
        
        Returns
        -------
        k_corrections: xp.array
        '''
        if self.band == 'W1':
            k_corr = -1*(4.44e-2+2.67*z+1.33*(z**2.)-1.59*(z**3.)) #From Maciej email
        elif self.band == 'K':
            # https://iopscience.iop.org/article/10.1086/322488/pdf 4th page lhs
            to_ret=-6.0*xp.log10(1+z)
            to_ret[z>0.3]=-6.0*xp.log10(1+0.3)
            k_corr=-6.0*xp.log10(1+z)
        elif self.band == 'bJ':
            # Fig 5 caption from https://arxiv.org/pdf/astro-ph/0111011.pdf
            # Note that these corrections also includes evolution corrections
            k_corr=(z+6*xp.power(z,2.))/(1+15.*xp.power(z,3.))
        return k_corr
        

class basic_redshift_rate(object):
    '''Super class for the redshift rate
    '''
    def evaluate(self,z):
        ''' Returns the rate
        
        Parameters
        ----------
        z: xp.array
            Redshift  
        '''
        return xp.exp(self.log_evaluate(z))
    
class powerlaw_rate(basic_redshift_rate):
    '''
    Class for a power-law redshift rate
    '''
    def __init__(self,gamma):
        self.gamma=gamma
    def log_evaluate(self,z):
        return self.gamma*xp.log1p(z)

class md_rate(basic_redshift_rate):
    '''
    Class for a MD like redshift rate
    '''
    def __init__(self,gamma,kappa,zp):
        self.gamma=gamma
        self.kappa=kappa
        self.zp=zp
    def log_evaluate(self,z):
        return xp.log1p(xp.power(1+self.zp,-self.gamma-self.kappa))+self.gamma*xp.log1p(z)-xp.log1p(xp.power((1+z)/(1+self.zp),self.gamma+self.kappa))


    
class basic_absM_rate(object):
    '''Super class for the redshift rate
    '''
    def evaluate(self,sch,M):
        ''' Returns the rate
        
        Parameters
        ----------
        sch: Schechter class
        M: xp.array
            Redshift  
        '''
        return xp.exp(self.log_evaluate(sch,M))

class log_powerlaw_absM_rate(basic_absM_rate):
    def __init__(self,epsilon):
        self.epsilon=epsilon
    def log_evaluate(self,sch,M):
        toret= self.epsilon*0.4*(sch.Mstarobs-M)*xp.log(10)
        # Note Galaxies fainter than the Schechter limit are assumed to have CBC rate 0.
        # Note Galaxies brighter are kept even if incosistent with Schechter limit 
        toret[(M>sch.Mmaxobs)]=-xp.inf
        return toret

