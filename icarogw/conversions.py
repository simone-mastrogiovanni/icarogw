from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn
import healpy as hp
from scipy.stats import gaussian_kde
from scipy.special import spence as PL
from tqdm import tqdm
from ligo.skymap.io.fits import read_sky_map
import astropy_healpix as ah
from astropy import units as u

def chirp_mass(m1,m2):
    '''
    Calculated the chirp mass

    Parameters
    ----------
    m1,m2: xp.array
        Masses in solar masses

    Returns
    -------
    Chirp mass
    '''

    xp = get_module_array(m1)
    return xp.power(m1*m2,3./5)/xp.power(m1+m2,1./5)

def mass_ratio(m1,m2):
    '''
    Computes the mass ratio (convention used is m1>m2 et q <1)

    Parameters
    ----------
    m1,m2: xp.array
        Masses in solar masses

    Returns
    -------
    mass ratio
    '''
    return m2/m1

def f_GW_ISCO(m1,m2):
    '''
    Returns the innermost stable circular orbit (0PN) approximation

    Parameters
    ----------
    m1,m2: xp.array
        Detector frame masses in solar masses
    
    Returns
    -------
    GW frequency corresponding to innermost stable circular orbit
    '''
    M=m1+m2
    f_ISCO = (2.20*(1/M))*10**3
    to_ret = f_ISCO*2
    return to_ret


class ligo_skymap(object):
    
    def __init__(self,skymapname):
        '''
        A class to store ligo.skymaps objects

        Parameters
        ----------

        skymapname: string
            path to the fits or fits.gz file released       
        '''
        
        self.table = read_sky_map(skymapname,distances=True,moc=True)
        self.intersected = False
        
    def intersect_EM_PE(self,ra,dec):
        '''
        Given a list or RA and DEC, it extracts from the skymap the associated distance and sky position probabilities
        Required for methods evaluate_3D_posterior_intersected and evaluate_3D_likelihood_intersected

        Parameters
        ----------
        ra: np.array
            Right ascension in radians
        dec: np.array
            Declination in radians
        '''
        
        xp = np

        if not self.intersected:
            ra*=u.rad
            dec*=u.rad
    
            # Taken from https://emfollow.docs.ligo.org/userguide/tutorial/multiorder_skymaps.html#probability-density-at-a-known-position
            level, ipix = ah.uniq_to_level_ipix(self.table['UNIQ'])
            px_area = ah.nside_to_pixel_area(ah.level_to_nside(level)).value # Pixel area in steradians
            nside = ah.level_to_nside(level)
    
            match_ipix = xp.zeros(len(ra),dtype=int)
            for i in range(len(ra)):
                mm = ah.lonlat_to_healpix(ra[i], dec[i], nside, order='nested') # Pixels corresponding to provided ra and dec
                ipp = np.flatnonzero(ipix == mm)[0]
                match_ipix[i] = ipix[ipp] # Pixel that 
     
            self.dl_means = xp.zeros(len(match_ipix))
            self.dl_sigmas = xp.zeros(len(match_ipix))
            self.sky_prob_rad2 = xp.zeros(len(match_ipix))
            self.pixels_area = xp.zeros(len(match_ipix))
    
            for i,pix in enumerate(match_ipix):
                idx = xp.where(ipix == pix)[0][0]
                self.dl_means[i] = self.table['DISTMU'][idx]
                self.dl_sigmas[i] = self.table['DISTSIGMA'][idx]
                self.sky_prob_rad2[i] = self.table['PROBDENSITY'][idx]
                self.pixels_area = px_area[i]
                
            self.intersected = True
        else:
            pass
        
    def evaluate_3D_posterior_intersected(self,dl):
        '''
        Returns the localization probability p(dL,RA,DEC)=p(RA,DEC)p(dL|RA,DEC). Requires intersect_EM_PE to be run before

        Parameters
        ----------
        dl: np.array
            Luminosity distance in Mpc

        Returns
        -------
        p(dL,RA,DEC)
        '''
        xp = np
        pdl_radec = xp.power(2*xp.pi*(self.dl_sigmas**2.),-2.)*xp.exp(-0.5*xp.power((dl-self.dl_means)/self.dl_sigmas,2.))
        prob = self.sky_prob_rad2 * pdl_radec    
        return prob
    
    def evaluate_3D_likelihood_intersected(self,dl):
        '''
        Returns the localization likelihood p(dL,RA,DEC)=p(RA,DEC)p(dL|RA,DEC)/prior(RA,DEC,dl). Requires intersect_EM_PE to be run before.
        the prior on dl is assumed to be dl^2 while the prior on the pixel is the inverse of its area (isotropic).

        Parameters
        ----------
        dl: np.array
            Luminosity distance in Mpc

        Returns
        -------
        Sky likelihood
        '''
        xp = np
        # The prior on the sky is 1/pixels_area (that is divided), the other term is the dl2 prior
        return self.evaluate_3D_posterior_intersected(dl)*self.pixels_area/xp.power(dl,2.)
    
    def evaluate_3D_posterior_likelihood(self,dl,ra,dec):
        '''
        Returns the localization posterior and likelihood.
        
        Parameters
        ----------
        dl: np.array
            Luminosity distance in Mpc
        ra: np.array
            Right ascension radians
        dec: np.array
            Declination radians

        Returns
        -------
        Posterior in [Mpc-1 sr-1] and likelihood.
        '''
        xp = np
        ra*=u.rad
        dec*=u.rad
        
        # Taken from https://emfollow.docs.ligo.org/userguide/tutorial/multiorder_skymaps.html#probability-density-at-a-known-position
        level, ipix = ah.uniq_to_level_ipix(self.table['UNIQ'])
        px_area = ah.nside_to_pixel_area(ah.level_to_nside(level)).value # Pixel area in steradians
        nside = ah.level_to_nside(level)
        
        match_ipix = xp.zeros(len(ra),dtype=int)
        for i in range(len(ra)):
            mm = ah.lonlat_to_healpix(ra[i], dec[i], nside, order='nested') # Pixels corresponding to provided ra and dec
            ipp = np.flatnonzero(ipix == mm)[0]
            match_ipix[i] = ipix[ipp] # Pixel that 

        dl_means = xp.zeros(len(match_ipix))
        dl_sigmas = xp.zeros(len(match_ipix))
        sky_prob_rad2 = xp.zeros(len(match_ipix))
        pixels_area = xp.zeros(len(match_ipix))

        for i,pix in enumerate(match_ipix):
            idx = xp.where(ipix == pix)[0][0]
            dl_means[i] = self.table['DISTMU'][idx]
            dl_sigmas[i] = self.table['DISTSIGMA'][idx]
            sky_prob_rad2[i] = self.table['PROBDENSITY'][idx]
            pixels_area = px_area[i]
            
        pdl_radec = xp.power(2*xp.pi*(dl_sigmas**2.),-2.)*xp.exp(-0.5*xp.power((dl-dl_means)/dl_sigmas,2.))
        prob = sky_prob_rad2 * pdl_radec 
        
        return prob, prob*pixels_area/xp.power(dl,2.)
        
    def sample_3d_space(self,Nsamp):
        '''
        Given the skymap, sample RA, DEC and dL from it.

        Parameters
        ----------
        Nsamp: int
            Number of samples to generate

        Returns
        -------
        dl: xp.array
            Luminosity distance in Mpc
        ra: xp.array
            Right ascension in radians
        dec: xp.array
            Declination in radians
        '''
        
        xp = np
        
        level, ipix = ah.uniq_to_level_ipix(self.table['UNIQ'])
        nside = ah.level_to_nside(level)    
        px_area = ah.nside_to_pixel_area(ah.level_to_nside(level)).value # Pixel area in steradians
        prob = self.table['PROBDENSITY']*px_area
        prob/=prob.sum()
        idx = xp.random.choice(len(ipix),size=Nsamp,replace=True,p=prob) 
        ra, dec = ah.healpix_to_lonlat(ipix[idx], nside[idx], order='nested')
        dl = xp.zeros(len(ra))
        
        match_ipix = ipix[idx]
       
        for i,pix in enumerate(match_ipix):
            idx = xp.where(ipix == pix)[0]
            dl_mean = self.table['DISTMU'][idx].value[0]
            dl_sigma = self.table['DISTSIGMA'][idx].value[0]

            if dl_mean<0:
                print('Skipping iteration {:d}, pixel {:d}, negative dl_mean {:.2f} Mpc'.format(i,pix,dl_mean))
                continue
            dldraw = -1
            while dldraw<=0:
                dldraw = xp.random.randn()*dl_sigma+dl_mean
            dl[i] = dldraw
        
        return dl, ra.rad, dec.rad

# LVK Reviewed
def cartestianspins2chis(s1x,s1y,s1z,s2x,s2y,s2z,q):
    '''
    Returns chi_1, chi_2, cosines with orbital angular momentum, chi_eff and chi_p 
    from cartesian spin components (z is on the orbital angular momentum).
    
    Parameters
    ----------
    s1x,s1y,s1z: xp.array
        Spin cartesian components of first object
    s2x,s2y,s2z: xp.array
        Spin cartesian components of secondary object
    q: xp.array
        mass ratio in (0,1]
    '''
    
    xp = get_module_array(s1x)
    
    chi_1 = xp.sqrt(s1x**2.+s1y**2.+s1z**2.)
    chi_2 = xp.sqrt(s2x**2.+s2y**2.+s2z**2.)
    cos_t_1, cos_t_2 = s1z/chi_1 , s2z/chi_2
    chi_eff = chi_eff_from_spins(chi_1, chi_2, cos_t_1, cos_t_2, q)
    chi_p = chi_p_from_spins(chi_1, chi_2, cos_t_1, cos_t_2, q)
    
    return chi_1, chi_2, cos_t_1, cos_t_2, chi_eff, chi_p

# LVK Reviewed
def Di(z):
    '''
    Function from https://github.com/tcallister/effective-spin-priors. Credit: T. Callister
    Derivation of equations can be found in arxiv:2104.09508
    
    Wrapper for the scipy implmentation of Spence's function.
    Note that we adhere to the Mathematica convention as detailed in:
    https://reference.wolfram.com/language/ref/PolyLog.html
    
    Parameters
    ----------
    z: complex scalar or array
  
    Returns
    -------
        Array equivalent to PolyLog[2,z], as defined by Mathematica
    '''

    return PL(1.-z+0j)

# LVK Reviewed
def chi_effective_prior_from_aligned_spins(q,aMax,xs):
    '''
    Function from https://github.com/tcallister/effective-spin-priors. Credit: T. Callister
    Derivation of equations can be found in arxiv:2104.09508
    
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, aligned component spin priors.
    
    Parameters
    ----------
    q: xp.array
        Mass ratio value (according to the convention q<1)
    aMax: float
        Maximum allowed dimensionless component spin magnitude
    xs: xp.array
        Chi_effective value or values at which we wish to compute prior
  
    Returns
    -------
    Array of prior values
    '''
    
    xp = get_module_array(q)

    # Ensure that `xs` is an array and take Absolute bolometric value
    origin_shape=xs.shape
    xs = xp.reshape(xs,-1)

    # Set up various piecewise cases
    pdfs = xp.zeros(xs.size)
    caseA = (xs>aMax*(1.-q)/(1.+q))*(xs<=aMax)
    caseB = (xs<-aMax*(1.-q)/(1.+q))*(xs>=-aMax)
    caseC = (xs>=-aMax*(1.-q)/(1.+q))*(xs<=aMax*(1.-q)/(1.+q))

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]

    pdfs[caseA] = (1.+q[caseA])**2.*(aMax-x_A)/(4.*q[caseA]*aMax**2)
    pdfs[caseB] = (1.+q[caseB])**2.*(aMax+x_B)/(4.*q[caseB]*aMax**2)
    pdfs[caseC] = (1.+q[caseC])/(2.*aMax)

    return pdfs.reshape(origin_shape)

# LVK Reviewed
def chi_effective_prior_from_isotropic_spins(q,aMax,xs):
    '''
    Function from https://github.com/tcallister/effective-spin-priors. Credit: T. Callister
    Derivation of equations can be found in arxiv:2104.09508
    
    Note: small fix for q with respect to the original version. The original version was not selecting
    cases for q, this caused the code to crush when provided with q and xs arrays.
    
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, isotropic component spin priors.
    
    Parameters
    ----------
    q: xp.array
        Mass ratio value (according to the convention q<1)
    aMax: float
        Maximum allowed dimensionless component spin magnitude
    xs: xp.array
        Chi_effective value or values at which we wish to compute prior
    
    Returns
    -------
    Array of prior values
    '''
    
    xp = get_module_array(q)

    # Ensure that `xs` is an array and take Absolute bolometric value
    origin_shape=xs.shape
    xs = xp.reshape(xp.abs(xs),-1)

    # Set up various piecewise cases
    pdfs = xp.ones(xs.size,dtype=complex)*(-1.)
    caseZ = (xs==0)
    caseA = (xs>0)*(xs<aMax*(1.-q)/(1.+q))*(xs<q*aMax/(1.+q))
    caseB = (xs<aMax*(1.-q)/(1.+q))*(xs>q*aMax/(1.+q))
    caseC = (xs>aMax*(1.-q)/(1.+q))*(xs<q*aMax/(1.+q))
    caseD = (xs>aMax*(1.-q)/(1.+q))*(xs<aMax/(1.+q))*(xs>=q*aMax/(1.+q))
    caseE = (xs>aMax*(1.-q)/(1.+q))*(xs>aMax/(1.+q))*(xs<aMax)
    caseF = (xs>=aMax)

    # Select relevant effective spins
    x_A, q_A = xs[caseA], q[caseA]
    x_B, q_B = xs[caseB], q[caseB]
    x_C, q_C = xs[caseC], q[caseC]
    x_D, q_D = xs[caseD], q[caseD]
    x_E, q_E = xs[caseE], q[caseE]
    q_Z = q[caseZ]

    pdfs[caseZ] = (1.+q_Z)/(2.*aMax)*(2.-xp.log(q_Z))

    pdfs[caseA] = (1.+q_A)/(4.*q_A*aMax**2)*(
                    q_A*aMax*(4.+2.*xp.log(aMax) - xp.log(q_A**2*aMax**2 - (1.+q_A)**2*x_A**2))
                    - 2.*(1.+q_A)*x_A*xp.arctanh((1.+q_A)*x_A/(q_A*aMax))
                    + (1.+q_A)*x_A*(Di(-q_A*aMax/((1.+q_A)*x_A)) - Di(q_A*aMax/((1.+q_A)*x_A)))
                    )

    pdfs[caseB] = (1.+q_B)/(4.*q_B*aMax**2)*(
                    4.*q_B*aMax
                    + 2.*q_B*aMax*xp.log(aMax)
                    - 2.*(1.+q_B)*x_B*xp.arctanh(q_B*aMax/((1.+q_B)*x_B))
                    - q_B*aMax*xp.log((1.+q_B)**2*x_B**2 - q_B**2*aMax**2)
                    + (1.+q_B)*x_B*(Di(-q_B*aMax/((1.+q_B)*x_B)) - Di(q_B*aMax/((1.+q_B)*x_B)))
                    )

    pdfs[caseC] = (1.+q_C)/(4.*q_C*aMax**2)*(
                    2.*(1.+q_C)*(aMax-x_C)
                    - (1.+q_C)*x_C*xp.log(aMax)**2.
                    + (aMax + (1.+q_C)*x_C*xp.log((1.+q_C)*x_C))*xp.log(q_C*aMax/(aMax-(1.+q_C)*x_C))
                    - (1.+q_C)*x_C*xp.log(aMax)*(2. + xp.log(q_C) - xp.log(aMax-(1.+q_C)*x_C))
                    + q_C*aMax*xp.log(aMax/(q_C*aMax-(1.+q_C)*x_C))
                    + (1.+q_C)*x_C*xp.log((aMax-(1.+q_C)*x_C)*(q_C*aMax-(1.+q_C)*x_C)/q_C)
                    + (1.+q_C)*x_C*(Di(1.-aMax/((1.+q_C)*x_C)) - Di(q_C*aMax/((1.+q_C)*x_C)))
                    )

    pdfs[caseD] = (1.+q_D)/(4.*q_D*aMax**2)*(
                    -x_D*xp.log(aMax)**2
                    + 2.*(1.+q_D)*(aMax-x_D)
                    + q_D*aMax*xp.log(aMax/((1.+q_D)*x_D-q_D*aMax))
                    + aMax*xp.log(q_D*aMax/(aMax-(1.+q_D)*x_D))
                    - x_D*xp.log(aMax)*(2.*(1.+q_D) - xp.log((1.+q_D)*x_D) - q_D*xp.log((1.+q_D)*x_D/aMax))
                    + (1.+q_D)*x_D*xp.log((-q_D*aMax+(1.+q_D)*x_D)*(aMax-(1.+q_D)*x_D)/q_D)
                    + (1.+q_D)*x_D*xp.log(aMax/((1.+q_D)*x_D))*xp.log((aMax-(1.+q_D)*x_D)/q_D)
                    + (1.+q_D)*x_D*(Di(1.-aMax/((1.+q_D)*x_D)) - Di(q_D*aMax/((1.+q_D)*x_D)))
                    )

    pdfs[caseE] = (1.+q_E)/(4.*q_E*aMax**2)*(
                    2.*(1.+q_E)*(aMax-x_E)
                    - (1.+q_E)*x_E*xp.log(aMax)**2
                    + xp.log(aMax)*(
                        aMax
                        -2.*(1.+q_E)*x_E
                        -(1.+q_E)*x_E*xp.log(q_E/((1.+q_E)*x_E-aMax))
                        )
                    - aMax*xp.log(((1.+q_E)*x_E-aMax)/q_E)
                    + (1.+q_E)*x_E*xp.log(((1.+q_E)*x_E-aMax)*((1.+q_E)*x_E-q_E*aMax)/q_E)
                    + (1.+q_E)*x_E*xp.log((1.+q_E)*x_E)*xp.log(q_E*aMax/((1.+q_E)*x_E-aMax))
                    - q_E*aMax*xp.log(((1.+q_E)*x_E-q_E*aMax)/aMax)
                    + (1.+q_E)*x_E*(Di(1.-aMax/((1.+q_E)*x_E)) - Di(q_E*aMax/((1.+q_E)*x_E)))
                    )

    pdfs[caseF] = 0.

    # Deal with spins on the boundary between cases
    if xp.any(pdfs==-1):
        boundary = (pdfs==-1)
        pdfs[boundary] = 0.5*(chi_effective_prior_from_isotropic_spins(q[boundary],aMax,xs[boundary]+1e-6)\
                        + chi_effective_prior_from_isotropic_spins(q[boundary],aMax,xs[boundary]-1e-6))

    return xp.real(pdfs).reshape(origin_shape)

# LVK Reviewed
def chi_p_prior_from_isotropic_spins(q,aMax,xs):
    '''
    Function from https://github.com/tcallister/effective-spin-priors. Credit: T. Callister
    Derivation of equations can be found in arxiv:2104.09508
    
    Function defining the conditional priors p(chi_p|q) corresponding to
    uniform, isotropic component spin priors.

    Parameters
    ----------
    q: xp.array
        Mass ratio value (according to the convention q<1)
    aMax: float
        Maximum allowed dimensionless component spin magnitude
    xs: xp.arry
        Chi_p value or values at which we wish to compute prior
    
    Returns:
    --------
    Array of prior values
    '''

    # Ensure that `xs` is an array and take Absolute bolometric value
    xp = get_module_array(q)
    origin_shape=xs.shape
    xs = xp.reshape(xs,-1)

    # Set up various piecewise cases
    pdfs = xp.zeros(xs.size)
    caseA = xs<q*aMax*(3.+4.*q)/(4.+3.*q)
    caseB = (xs>=q*aMax*(3.+4.*q)/(4.+3.*q))*(xs<aMax)

    # Select relevant effective spins
    x_A, q_A = xs[caseA], q[caseA]
    x_B, q_B = xs[caseB], q[caseB]

    pdfs[caseA] = (1./(aMax**2*q_A))*((4.+3.*q_A)/(3.+4.*q_A))*(
                    xp.arccos((4.+3.*q_A)*x_A/((3.+4.*q_A)*q*aMax))*(
                        aMax
                        - xp.sqrt(aMax**2-x_A**2)
                        + x_A*xp.arccos(x_A/aMax)
                        )
                    + xp.arccos(x_A/aMax)*(
                        aMax*q_A*(3.+4.*q_A)/(4.+3.*q_A)
                        - xp.sqrt(aMax**2*q_A**2*((3.+4.*q_A)/(4.+3.*q_A))**2 - x_A**2)
                        + x_A*xp.arccos((4.+3.*q_A)*x_A/((3.+4.*q_A)*aMax*q_A))
                        )
                    )

    pdfs[caseB] = (1./aMax)*xp.arccos(x_B/aMax)

    return pdfs.reshape(origin_shape)

# LVK Reviewed
def joint_prior_from_isotropic_spins(q,aMax,xeffs,xps,ndraws=10000,bw_method='scott'):
    '''
    Function from https://github.com/tcallister/effective-spin-priors. Credit: T. Callister
    Derivation of equations can be found in arxiv:2104.09508
    
    Function to calculate the conditional priors p(xp|xeff,q) on a set of {xp,xeff,q} posterior samples.
    
    Parameters
    ----------
    q: xp.array
        Mass ratio
    aMax: float
        Maximimum spin magnitude considered
    xeffs: xp.array
        Effective inspiral spin samples
    xps: xp.array
        Effective precessing spin values
    ndraws: int
        Number of draws from the component spin priors used in numerically building interpolant
    
    Returns
    -------
    
    Array of priors on xp, conditioned on given effective inspiral spins and mass ratios
    '''
    xp=get_module_array(xeffs)

    # Convert to arrays for safety
    origin_shape=xeffs.shape
    xeffs = xp.reshape(xeffs,-1)
    xps = xp.reshape(xps,-1)

    # Compute marginal prior on xeff, conditional prior on xp, and multiply to get joint prior!
    p_chi_eff = chi_effective_prior_from_isotropic_spins(q,aMax,xeffs)
    p_chi_p_given_chi_eff = xp.zeros(len(p_chi_eff))
    
    for i in tqdm(range(len(p_chi_eff)),desc='Calculating p(chi_p|chi_eff,q)'):
        p_chi_p_given_chi_eff[i] = chi_p_prior_given_chi_eff_q(q[i],aMax,xeffs[i],xps[i],ndraws,bw_method)
    joint_p_chi_p_chi_eff = p_chi_eff*p_chi_p_given_chi_eff

    return joint_p_chi_p_chi_eff.reshape(origin_shape)

# LVK Reviewed
def chi_p_prior_given_chi_eff_q(q,aMax,xeff,xp,ndraws=10000,bw_method='scott'):
    '''
    Function from https://github.com/tcallister/effective-spin-priors. Credit: T. Callister
    Derivation of equations can be found in arxiv:2104.09508
    
    Function to calculate the conditional prior p(xp|xeff,q) on a single {xp,xeff,q} posterior sample.
    Called by `joint_prior_from_isotropic_spins`.
    
    Parameters
    ----------
    q: xp.array
        Single posterior mass ratio sample
    aMax: float
        Maximimum spin magnitude considered
    xeff: xp.array
        Single effective inspiral spin sample
    xp: xp.array
        Single effective precessing spin value
    ndraws: int
        Number of draws from the component spin priors used in numerically building interpolant
    
    Returns
    -------
    Prior on xp, conditioned on given effective inspiral spin and mass ratio
    '''
    
    xp=get_module_array(q)

    # Draw random spin magnitudes.
    # Note that, given a fixed chi_eff, a1 can be no larger than (1+q)*chi_eff,
    # and a2 can be no larger than (1+q)*chi_eff/q
    a1 = xp.random.random(ndraws)*aMax
    a2 = xp.random.random(ndraws)*aMax

    # Draw random tilts for spin 2
    cost2 = 2.*xp.random.random(ndraws)-1.

    # Finally, given our conditional value for chi_eff, we can solve for cost1
    # Note, though, that we still must require that the implied value of cost1 be *physical*
    cost1 = (xeff*(1.+q) - q*a2*cost2)/a1

    # While any cost1 values remain unphysical, redraw a1, a2, and cost2, and recompute
    # Repeat as necessary
    while xp.any(cost1<-1) or xp.any(cost1>1):
        to_replace = xp.where((cost1<-1) | (cost1>1))[0]
        a1[to_replace] = xp.random.random(to_replace.size)*aMax
        a2[to_replace] = xp.random.random(to_replace.size)*aMax
        cost2[to_replace] = 2.*xp.random.random(to_replace.size)-1.
        cost1 = (xeff*(1.+q) - q*a2*cost2)/a1

    # Compute precessing spins and corresponding weights, build KDE
    # See `Joint-ChiEff-ChiP-Prior.ipynb` for a discussion of these weights
    Xp_draws = chi_p_from_spins(a1,a2,cost1,cost2,q)
    jacobian_weights = (1.+q)/a1
    prior_kde = gaussian_kde(Xp_draws,weights=jacobian_weights,bw_method=bw_method)

    # Compute maximum chi_p
    if (1.+q)*xp.abs(xeff)/q<aMax:
        max_Xp = aMax
    else:
        max_Xp = xp.sqrt(aMax**2 - ((1.+q)*xp.abs(xeff)-q)**2.)

    # Set up a grid slightly inside (0,max chi_p) and evaluate KDE
    reference_grid = xp.linspace(0.05*max_Xp,0.95*max_Xp,50)
    reference_vals = prior_kde(reference_grid)

    # Manually prepend/append zeros at the boundaries
    reference_grid = xp.concatenate([[0],reference_grid,[max_Xp]])
    reference_vals = xp.concatenate([[0],reference_vals,[0]])
    norm_constant = xp.trapz(reference_vals,reference_grid)

    # Interpolate!
    p_chi_p = xp.interp(xp,reference_grid,reference_vals/norm_constant)
    return p_chi_p

# LVK Reviewed
def chi_eff_from_spins(chi1, chi2, cos1, cos2, q):
    '''
    Function to tranform the usual spins parameters chi, chi2, cos1, cos2
    into the the effective spin parameters chi_eff.
    
    Parameters
    ----------
    chi1: primary dimensionless spin magnitude
    chi2: secondary spin magnitude
    cos1: cosine of the primary tilt angle
    cos2: cosine of the secondary tilt angle
    q: mass ratio   \propto (m2/m1)
    
    Returns
    -------
    chi effective
    '''

    to_ret = (chi1*cos1 + q*chi2*cos2)/(1.+q)
    return to_ret

# LVK Reviewed
def chi_p_from_spins(chi1, chi2, cos1, cos2, q):
    '''
    Function to tranform the usual spins parameters chi, chi2, cos1, cos2
    into the the precessing spin parameters  chi_p
    
    Parameters
    ----------
    chi1: primary dimensionless spin magnitude
    chi2: secondary spin magnitude
    cos1: cosine of the primary tilt angle
    cos2: cosine of the secondary tilt angle
    q: mass ratio   \propto (m2/m1)
    
    Returns
    -------
    chi p
    '''
    xp = get_module_array(chi1)
    sin1 = xp.sqrt(1.-cos1**2)
    sin2 = xp.sqrt(1.-cos2**2)

    to_ret = xp.maximum(chi1*sin1, ((4*q+3)/(3*q+4))*q*chi2*sin2)
    
    return to_ret

# LVK Reviewed
def radec2skymap(ra,dec,nside):
    '''
    Converts RA and DEC samples to a skymap with normalized probability, i.e. integral of skymap in dArea =1
    
    Parameters
    ----------
    ra, dec: xp.array
        arrays with RA and DEC in radians
    nside: int
        nside for healpy
    
    Returns
    -------
    counts_maps: xp.array
        Healpy array with skymap
    dOmega_sterad: float
        Area in steradians of the sky cell
    '''

    npixels=hp.nside2npix(nside)
    xp=get_module_array(ra)
    dOmega_sterad=4*xp.pi/npixels
    dOmega_deg2=xp.power(180/xp.pi,2.)*dOmega_sterad
    indices = radec2indeces(ra,dec,nside)
    counts_map = xp.zeros(npixels)
    for indx in range(npixels):
        ind=xp.where(indices==indx)[0]
        counts_map[indx]=len(ind)
        if ind.size==0:
            continue
    counts_map/=(len(ra)*dOmega_sterad)
    return counts_map, dOmega_sterad

# LVK Reviewed
def L2M(L):
    '''
    Converts Luminosity in Watt to Absolute bolometric magnitude
    
    Parameters
    ----------
    L: xp.array
        Luminosity in Watt
    
    Returns
    -------
    M: xp.array
        Absolute bolometric magnitude
    '''
    xp=get_module_array(L)
    # From Resolution B2 proposed by IAU, see e.g. Pag. 2, Eq. 2 of https://www.iau.org/static/resolutions/IAU2015_English.pdf
    return -2.5*xp.log10(L)+71.197425

# LVK Reviewed
def M2L(M):
    '''
    Converts Absolute bolometric magnitude to Luminosity in Watt 
    
    Parameters
    ----------
    M: xp.array
        Absolute bolometric magnitude
    
    Returns
    -------
    L: xp.array
        Luminosity in Watt
    '''
    xp=get_module_array(M)
    # From Pag. 2, Eq. 1-2 of https://www.iau.org/static/resolutions/IAU2015_English.pdf
    return 3.0128e28*xp.power(10.,-0.4*M)

# LVK Reviewed
def radec2indeces(ra,dec,nside):
    '''
    Converts RA and DEC to healpy indeces
    
    Parameters
    ----------
    ra, dec: np.array
        arrays with RA and DEC in radians
    nside: int
        nside for healpy
    
    Returns
    -------
    healpy indeces as numpy array
    '''
    theta = np.pi/2.0 - dec
    phi = ra
    return hp.ang2pix(nside, theta, phi)

# LVK Reviewed
def indices2radec(indices,nside):
    '''
    Converts healpy indeces to RA DEC
    
    Parameters
    ----------
    indices: np.array
        Healpy indices
    nside: int
        nside for healpy
    
    Returns
    -------
    ra, dec: np.array
        arrays with RA and DEC in radians
    '''
    
    theta,phi= hp.pix2ang(nside,indices)
    return phi, np.pi/2.0-theta

# LVK Reviewed
def M2m(M,dl,kcorr):
    '''
    Converts Absolute bolometric magnitude to apparent magnitude
    
    Parameters
    ----------
    M: xp.array
        Absolute bolometric magnitude
    dl: xp.array
        Luminosity distance in Mpc
    kcorr: xp.array
        K-correction, we suggest to use the kcorr class
    
    Returns
    -------
    m: xp.array
        Apparent magnitude
    '''
    xp=get_module_array(M)
    # Note that the distance is Mpc here. See Eq. 2 of https://arxiv.org/abs/astro-ph/0210394
    dist_modulus=5*xp.log10(dl)+25.
    return M+dist_modulus+kcorr

# LVK Reviewed
def m2M(m,dl,kcorr):
    '''
    Converts apparent magnitude to Absolute bolometric magnitude
    
    Parameters
    ----------
    m: xp.array
        apparebt magnitude
    dl: xp.array
        Luminosity distance in Mpc
    kcorr: xp.array
        K-correction, we suggest to use the kcorr class
    
    Returns
    -------
    M: xp.array
        Absolute bolometric magnitude
    '''
    xp=get_module_array(m)    
    # Note that the distance is Mpc here. See Eq. 2 of https://arxiv.org/abs/astro-ph/0210394
    dist_modulus=5*xp.log10(dl)+25.
    return m-dist_modulus-kcorr

# LVK Reviewed
def source2detector(mass1_source,mass2_source,z,cosmology):
    '''
    Converts from Source frame to detector frame quantities
    
    Parameters
    ----------
    mass1_source: xp.array
        Source mass of the primary
    mass2_source: xp.array
        Source mass of the secondary
    z: xp.array
        Redshift of the object
    cosmology: class
        Cosmology class from the cosmology module
    
    Returns
    -------
    Detector frame masses and luminosity distance in Mpc
    '''
    return mass1_source*(1+z),mass2_source*(1+z),cosmology.z2dl(z)

# LVK Reviewed
def detector2source(mass1,mass2,dl,cosmology):
    '''
    Converts from Source frame to detector frame quantities
    
    Parameters
    ----------
    mass1: xp.array
        Detector mass of the primary
    mass2: xp.array
        Detector mass of the secondary
    dl: xp.array
        Luminosity distnce in Mpc
    cosmology: class
        Cosmology class from the cosmology module
    
    Returns
    -------
    Source frame masses and redshift
    '''
    
    z=cosmology.dl2z(dl)
    return mass1/(1+z),mass2/(1+z),z

# LVK Reviewed
def detector2source_jacobian(z, cosmology):
    '''
    Calculates the detector frame to source frame Jacobian d_det/d_sour

    Parameters
    ----------
    z: xp. arrays
        Redshift
    cosmo:  class from the cosmology module
        Cosmology class from the cosmology module
    '''
    xp=get_module_array(z)
    return xp.abs(xp.power(1+z,2.)*cosmology.ddl_by_dz_at_z(z))

# LVK Reviewed
def source2detector_jacobian(z, cosmology):
    '''
    Calculates the detector frame to source frame Jacobian d_sour/d_det

    Parameters
    ----------
    z: xp. arrays
        Redshift
    cosmo:  class from the cosmology module
        Cosmology class from the cosmology module
    '''
    return 1./detector2source_jacobian(z,cosmology)
    
