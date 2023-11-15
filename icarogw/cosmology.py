from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn, is_there_cupy
from icarogw import cupy_pal
from scipy.integrate import cumtrapz
import mpmath

COST_C= 299792.458 # Speed of light in km/s

# LVK Reviewed
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
        self.log10_z_cpu=np.log10(self.z_cpu)
        
        if is_there_cupy():
            self.z_gpu=cupy_pal.cp.logspace(-6,np.log10(self.zmax),2500)
            self.log10_z_gpu=cupy_pal.cp.log10(self.z_gpu)
        
    def _checkz(self,z):
        smin,smax=z.min(),z.max()
        if (smin<1e-6) | (smax>self.zmax):
            raise ValueError('Redshift provided not in range 1e-6<z<{:.2f}, zmin = {:f}, zmax = {:f}'.format(self.zmax,smin,smax))
            
    def _checkdl(self,dl):
        xp = get_module_array(dl)
        dlmin,dlmax=dl.min(),dl.max()
        
        if iscupy(dl):
            log10_dl_at_z = self.log10_dl_at_z_gpu
        else:
            log10_dl_at_z = self.log10_dl_at_z_cpu
        
        dlmingrid,dlmaxgrid=xp.power(10.,log10_dl_at_z.min()),xp.power(10.,log10_dl_at_z.max())

        if (dlmin<dlmingrid) | (dlmax>dlmaxgrid):
            raise ValueError('Luminosity provided not in range {:f}<dl<{:f} Mpc, dlmin = {:f} Mpc, dlmax = {:f} Mpc'.format(dlmingrid,dlmaxgrid,
                                                                                                                           dlmin,dlmax))
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
        self._checkz(z)
        origin = z.shape
        xp = get_module_array(z)
        
        if iscupy(z):
            log10_z = self.log10_z_gpu
            log10_dl_at_z = self.log10_dl_at_z_gpu
        else:
            log10_z = self.log10_z_cpu
            log10_dl_at_z = self.log10_dl_at_z_cpu
        
        ravelled=xp.ravel(xp.log10(z))
        interpo=xp.interp(ravelled,log10_z,log10_dl_at_z)
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
        self._checkz(z)
        origin=z.shape
        xp = get_module_array(z)
        
        if iscupy(z):
            log10_z = self.log10_z_gpu
            log10_Vc = self.log10_Vc_gpu
        else:
            log10_z = self.log10_z_cpu
            log10_Vc = self.log10_Vc_cpu
        
        
        ravelled=xp.ravel(xp.log10(z))
        interpo=xp.interp(ravelled,log10_z,log10_Vc)
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
        self._checkdl(dl)
        origin=dl.shape
        xp = get_module_array(dl)
        
        if iscupy(dl):
            log10_z = self.log10_z_gpu
            log10_dl_at_z = self.log10_dl_at_z_gpu
        else:
            log10_z = self.log10_z_cpu
            log10_dl_at_z = self.log10_dl_at_z_cpu
        
        ravelled=xp.ravel(xp.log10(dl))
        interpo=xp.interp(ravelled,log10_dl_at_z,log10_z)
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
        self._checkz(z)
        origin=z.shape
        xp = get_module_array(z)
        
        if iscupy(z):
            log10_z = self.log10_z_gpu
            log10_dVc_dzdOmega = self.log10_dVc_dzdOmega_gpu
        else:
            log10_z = self.log10_z_cpu
            log10_dVc_dzdOmega = self.log10_dVc_dzdOmega_cpu
        
        ravelled=xp.ravel(xp.log10(z))
        interpo=xp.interp(ravelled,log10_z,log10_dVc_dzdOmega)
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
        self._checkz(z)
        origin=z.shape
        xp = get_module_array(z)
        
        if iscupy(z):
            log10_z = self.log10_z_gpu
            log10_ddl_by_dz = self.log10_ddl_by_dz_gpu
        else:
            log10_z = self.log10_z_cpu
            log10_ddl_by_dz = self.log10_ddl_by_dz_cpu
        
        ravelled=xp.ravel(xp.log10(z))
        interpo=xp.interp(ravelled,log10_z,log10_ddl_by_dz)
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
        self._checkz(np.array([zmin,zmax]))
        zproxy=np.linspace(zmin,zmax,10000)
        prob=self.dVc_by_dzdOmega_at_z(zproxy)
        cdf=np.cumsum(prob)/prob.sum()
        cdf[0]=0.
        cdf_samps=np.random.rand(Nsamp)
        return np.interp(cdf_samps,cdf,zproxy)
    
# LVK Reviewed
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
        
        self.log10_dVc_dzdOmega_cpu=np.log10(astropy_cosmo.differential_comoving_volume(self.z_cpu).value)-9. # Conversion from Mpc to Gpc
        self.log10_Vc_cpu=np.log10(astropy_cosmo.comoving_volume(self.z_cpu).value)-9. # Conversion to Gpc
        self.log10_dl_at_z_cpu=np.log10(astropy_cosmo.luminosity_distance(self.z_cpu).value)
        self.log10_ddl_by_dz_cpu=np.log10((np.power(10.,self.log10_dl_at_z_cpu)/(1.+self.z_cpu))+COST_C*(1.+self.z_cpu)/astropy_cosmo.H(self.z_cpu).value)
        
        if is_there_cupy():
            
            self.log10_dVc_dzdOmega_gpu=np2cp(self.log10_dVc_dzdOmega_cpu)
            self.log10_Vc_gpu=np2cp(self.log10_Vc_cpu)
            self.log10_dl_at_z_gpu=np2cp(self.log10_dl_at_z_cpu)
            self.log10_ddl_by_dz_gpu=np2cp(self.log10_ddl_by_dz_cpu)

# LVK Reviewed
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
        dlem = np.power(10.,self.log10_dl_at_z_cpu)
        dlbydz_em = np.power(10.,self.log10_ddl_by_dz_cpu)
        
        self.log10_dl_at_z_cpu=np.log10(dlem*np.power(1+np.power(dlem/((1.+self.z_cpu)*Rc),n),(D-4.)/(2*n)))
        Afa=1.+np.power(dlem/((1.+self.z_cpu)*Rc),n)
        expo=(D-4.)/(2.*n)
        # We put the absolute value for the Jacobian (because this is needed for probabilities)
        self.log10_ddl_by_dz_cpu=np.log10(np.abs(np.power(Afa,expo)*(dlbydz_em+np.power(dlem/Rc,n)*(expo*n/Afa)*\
            (dlbydz_em/np.power(1.+self.z_cpu,n)-dlem/np.power(1.+self.z_cpu,n+1.)))))
        
        if is_there_cupy():
            self.log10_dl_at_z_gpu=np2cp(self.log10_dl_at_z_cpu)
            self.log10_ddl_by_dz_gpu=np2cp(self.log10_ddl_by_dz_cpu)

# LVK Reviewed
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
        dlem = np.power(10.,self.log10_dl_at_z_cpu)
        dlbydz_em = np.power(10.,self.log10_ddl_by_dz_cpu)
        # Implementation of the running Planck mass model general case as described in the overleaf https://www.overleaf.com/project/62330c2859bb3c2a5982c2b6
        # Define array for numerical integration
        ZforI=np.append(self.z_cpu,self.z_cpu[-1]-self.z_cpu[-2])
        ZforI=np.append(0.,ZforI)
        Zhalfbin=(ZforI[:-1:]+ZforI[1::])*0.5
        Integrandhalfin=1./((1+Zhalfbin)*np.power(astropy_cosmo.efunc(Zhalfbin),2.))
        Integrand=1./((1+self.z_cpu)*np.power(astropy_cosmo.efunc(self.z_cpu),2.))
        Integral=cumtrapz(Integrandhalfin,Zhalfbin)
        self.log10_dl_at_z_cpu=np.log10(dlem*np.exp((0.5*cM)*Integral))
        exp_factor = np.exp(0.5*cM*Integral)
        # We put the absolute value for the Jacobian (because this is needed for probabilities)
        self.log10_ddl_by_dz_cpu=np.log10(np.abs(exp_factor*dlbydz_em+0.5*dlem*cM*exp_factor*Integrand))
        
        if is_there_cupy():
            self.log10_dl_at_z_gpu=np2cp(self.log10_dl_at_z_cpu)
            self.log10_ddl_by_dz_gpu=np2cp(self.log10_ddl_by_dz_cpu)
        
# LVK Reviewed        
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
        dlem = np.power(10.,self.log10_dl_at_z_cpu)
        dlbydz_em = np.power(10.,self.log10_ddl_by_dz_cpu)
        self.log10_dl_at_z_cpu=np.log10(dlem*(Xi0+(1.-Xi0)*np.power(1.+self.z_cpu,-n)))
        # We put the absolute value for the Jacobian (because this is needed for probabilities)
        self.log10_ddl_by_dz_cpu=np.log10(np.abs(dlbydz_em*(Xi0+(1.-Xi0)*np.power(1+self.z_cpu,-n))-dlem*(1.-Xi0)*n*np.power(1+self.z_cpu,-n-1)))
        
        if is_there_cupy():
            self.log10_dl_at_z_gpu=np2cp(self.log10_dl_at_z_cpu)
            self.log10_ddl_by_dz_gpu=np2cp(self.log10_ddl_by_dz_cpu)

# LVK Reviewed
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
        dlem = np.power(10.,self.log10_dl_at_z_cpu)
        dlbydz_em = np.power(10.,self.log10_ddl_by_dz_cpu)
        self.log10_dl_at_z_cpu=np.log10(dlem*(1.+alphalog_1*np.log1p(self.z_cpu)+\
                                alphalog_2*(np.log1p(self.z_cpu)**2.)+\
                                alphalog_3*(np.log1p(self.z_cpu)**3.)))        
        part1 = dlbydz_em*(1.+alphalog_1*np.log1p(self.z_cpu)+ \
                                        alphalog_2*(np.log1p(self.z_cpu))**2.+\
                                        alphalog_3*(np.log1p(self.z_cpu))**3.)
        part2 = (dlem/(1+self.z_cpu))*(alphalog_1+alphalog_2*2.*np.log1p(self.z_cpu)+\
                      alphalog_3*3.*(np.log1p(self.z_cpu)**2.))
        # We put the absolute value for the Jacobian (because this is needed for probabilities)
        self.log10_ddl_by_dz_cpu=np.log10(np.abs(part1 + part2))
        
        if is_there_cupy():
            self.log10_dl_at_z_gpu=np2cp(self.log10_dl_at_z_cpu)
            self.log10_ddl_by_dz_gpu=np2cp(self.log10_ddl_by_dz_cpu)

# LVK Reviewed    
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
        xp = get_module_array(M)
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
        xp = get_module_array(M)
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
        xp = get_module_array(M)
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
        xp = get_module_array(M)
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
        sarray=np.linspace(self.Mminobs,self.Mmaxobs,10000)
        cdfeval=np.cumsum(self.pdf(sarray))/self.pdf(sarray).sum()
        cdfeval[0]=0.
        randomcdf=np.random.rand(N)
        return np.interp(randomcdf,cdfeval,sarray,left=self.Mminobs,right=self.Mmaxobs)
    
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
        
        self.effective_density_interpolant_cpu=self.effective_density_interpolant[::-1]
        self.xvector_interpolant_cpu=self.Mstar-Mvector_interpolant[::-1]
        
        
        if is_there_cupy():
            self.effective_density_interpolant_gpu=np2cp(self.effective_density_interpolant[::-1])
            self.xvector_interpolant_gpu=np2cp(self.Mstar-Mvector_interpolant[::-1])
        
    def background_effective_galaxy_density(self,Mthr):
        '''Returns the effective galaxy density, i.e. dN_{gal,eff}/dVc, the effective number is given by the luminosity weights.
        This is Eq. 2.37 on the Overleaf documentation
        
        Parameters
        ----------
        Mthr: xp.array
            Absolute magnitude threshold (faint) used to compute the integral
        '''
        
        origin=Mthr.shape
        xp = get_module_array(Mthr)
        ravelled=xp.ravel(self.Mstarobs-Mthr)
        # Schecter function is 0 outside intervals that's why we set limit on boundaries
        
        if iscupy(Mthr):
            xvector_interpolant=self.xvector_interpolant_gpu
            effective_density_interpolant=self.effective_density_interpolant_gpu
        else:
            xvector_interpolant=self.xvector_interpolant_cpu
            effective_density_interpolant=self.effective_density_interpolant_cpu
            
        outp=self.phistarobs*xp.interp(ravelled,xvector_interpolant,effective_density_interpolant
                           ,left=effective_density_interpolant[0],right=effective_density_interpolant[-1])
        return xp.reshape(outp,origin)

# LVK Reviewed
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
        xp = get_module_array(z)
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

# LVK Reviewed
class basic_redshift_rate(object):
    '''
    Super class for the redshift rate
    '''
    def evaluate(self,z):
        ''' Returns the rate

        Parameters
        ----------
        z: xp.array
            Redshift  
        '''
        xp = get_module_array(z)
        return xp.exp(self.log_evaluate(z))

# LVK Reviewed
class powerlaw_rate(basic_redshift_rate):
    '''
    Class for a power-law redshift rate
    '''
    def __init__(self,gamma):
        self.gamma=gamma
    def log_evaluate(self,z):
        xp = get_module_array(z)
        return self.gamma*xp.log1p(z)

# LVK Reviewed
class md_rate(basic_redshift_rate):
    '''
    Class for a MD like redshift rate
    '''
    def __init__(self,gamma,kappa,zp):
        self.gamma=gamma
        self.kappa=kappa
        self.zp=zp
    def log_evaluate(self,z):
        xp = get_module_array(z)
        return xp.log1p(xp.power(1+self.zp,-self.gamma-self.kappa))+self.gamma*xp.log1p(z)-xp.log1p(xp.power((1+z)/(1+self.zp),self.gamma+self.kappa))


# LVK Reviewed
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
        xp = get_module_array(M)
        return xp.exp(self.log_evaluate(sch,M))

# LVK Reviewed
class log_powerlaw_absM_rate(basic_absM_rate):
    def __init__(self,epsilon):
        self.epsilon=epsilon
    def log_evaluate(self,sch,M):
        xp = get_module_array(M)
        toret= self.epsilon*0.4*(sch.Mstarobs-M)*xp.log(10)
        # Note Galaxies fainter than the Schechter limit are assumed to have CBC rate 0.
        # Note Galaxies brighter are kept even if incosistent with Schechter limit 
        toret[(M>sch.Mmaxobs)]=-xp.inf
        return toret

