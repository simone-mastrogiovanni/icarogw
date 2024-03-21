import numpy as np
import matplotlib.pylab as plt
import astropy.constants

def v(Mtot,f):
    '''
    This function computes the v factor (specify what it is)
    
    Parameters
    ----------
    Mtot: float
        Total mass of the binay in solar masses
    f: array
        Frequency array
    '''
    MsunToSec = astropy.constants.M_sun.value*astropy.constants.G.value/np.power(astropy.constants.c.value,3.)
    return np.array([(np.pi*MsunToSec*f*Mtot)**(1./3.), (np.pi*MsunToSec*f*Mtot)**(2./3.), (np.pi*MsunToSec*f*Mtot)])

def dEdf(Mtot,freqs,eta=0.25,inspiralOnly=False,PN=True,chi=None):

    """
    Function to compute the energy spectrum radiated by a CBC. Taken from (https://ui.adsabs.harvard.edu/abs/2023arXiv231017625T/abstract)
    
    INPUTS
    Mtot: Total mass in units of Msun
    freqs: Array of frequencies at which we want to evaluate dEdf
    eta: Reduced mass ratio. Defaults to 0.25 (equal mass)
    inspiralOnly: If True, will return only energy radiated through inspiral
    """

    Msun = astropy.constants.M_sun.value
    c= astropy.constants.c.value
    G= astropy.constants.G.value
    MsunToSec = astropy.constants.M_sun.value*astropy.constants.G.value/np.power(astropy.constants.c.value,3.)


    if chi is None:
        chi = 0.

    # Initialize energy density
    dEdf_spectrum = np.zeros_like(freqs)

    if inspiralOnly:

        # If inspiral only (used for BNS), cut off at the ISCO
        fMerge = 2.*c**3./(6.*np.sqrt(6.)*2.*np.pi*G*Mtot*Msun)
        inspiral = freqs<fMerge
        dEdf_spectrum[inspiral] = np.power(freqs[inspiral],-1./3.)

    else:

        if PN:

            # Waveform model from Ajith+ 2011 (10.1103/PhysRevLett.106.241101)

            # PN corrections to break frequencies bounding different waveform regimes
            # See Eq. 2 and Table 1
            eta_arr = np.array([eta,eta*eta,eta*eta*eta])
            chi_arr = np.array([1,chi,chi*chi]).T
            fM_corrections = np.array([[0.6437,0.827,-0.2706],[-0.05822,-3.935,0.],[-7.092,0.,0.]])
            fR_corrections = np.array([[0.1469,-0.1228,-0.02609],[-0.0249,0.1701,0.],[2.325,0.,0.]])
            fC_corrections = np.array([[-0.1331,-0.08172,0.1451],[-0.2714,0.1279,0.],[4.922,0.,0.]])
            sig_corrections = np.array([[-0.4098,-0.03523,0.1008],[1.829,-0.02017,0.],[-2.87,0.,0.]])

            # Define frequencies
            # See Eq. 2 and Table 1
            fMerge = (1. - 4.455*(1.-chi)**0.217 + 3.521*(1.-chi)**0.26 + eta_arr.dot(fM_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
            fRing = (0.5 - 0.315*(1.-chi)**0.3 + eta_arr.dot(fR_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
            fCut = (0.3236 + 0.04894*chi + 0.01346*chi*chi + eta_arr.dot(fC_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
            sigma = (0.25*(1.-chi)**0.45 - 0.1575*(1.-chi)**0.75 + eta_arr.dot(sig_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)

            # Identify piecewise components
            inspiral = freqs<fMerge
            merger = (freqs>=fMerge)*(freqs<fRing)
            ringdown = (freqs>=fRing)*(freqs<fCut)

            # Define PN amplitude corrections
            # See Eq. 1 and following text
            alpha = np.array([0., -323./224. + 451.*eta/168., (27./8.-11.*eta/6.)*chi])
            eps = np.array([1.4547*chi-1.8897, -1.8153*chi+1.6557, 0.])
            vs = v(Mtot,freqs)

            # Compute multiplicative scale factors to enforce continuity of dEdf across boundaries
            # Note that w_m and w_r are the ratios (inspiral/merger) and (merger/ringdown), as defined below
            v_m = v(Mtot,fMerge)
            v_r = v(Mtot,fRing)
            w_m = np.power(fMerge,-1./3.)*np.power(1.+alpha.dot(v_m),2.)/(np.power(fMerge,2./3.)*np.power(1.+eps.dot(v_m),2.)/fMerge)
            w_r = (w_m*np.power(fRing,2./3.)*np.power(1.+eps.dot(v_r),2.)/fMerge)/(np.square(fRing)/(fMerge*fRing**(4./3.)))

            # Energy spectrum --> https://arxiv.org/abs/2306.09861
            dEdf_spectrum[inspiral] = np.power(freqs[inspiral],-1./3.)*np.power(1.+alpha.dot(vs[:,inspiral]),2.)
            dEdf_spectrum[merger] = w_m*np.power(freqs[merger],2./3.)*np.power(1.+eps.dot(vs[:,merger]),2.)/fMerge
            dEdf_spectrum[ringdown] = w_r*np.square(freqs[ringdown]/(1.+np.square((freqs[ringdown]-fRing)/(sigma/2.))))/(fMerge*fRing**(4./3.))

        else:

            # Waveform model from Ajith+ 2008 (10.1103/PhysRevD.77.104017)
            # Define IMR parameters
            # See Eq. 4.19 and Table 1
            fMerge = (0.29740*eta**2. + 0.044810*eta + 0.095560)/(np.pi*Mtot*MsunToSec)
            fRing = (0.59411*eta**2. + 0.089794*eta + 0.19111)/(np.pi*Mtot*MsunToSec)
            fCut = (0.84845*eta**2. + 0.12828*eta + 0.27299)/(np.pi*Mtot*MsunToSec)
            sigma = (0.50801*eta**2. + 0.077515*eta + 0.022369)/(np.pi*Mtot*MsunToSec)

            # Identify piecewise components
            inspiral = freqs<fMerge
            merger = (freqs>=fMerge)*(freqs<fRing)
            ringdown = (freqs>=fRing)*(freqs<fCut)

            # Energy spectrum
            dEdf_spectrum[inspiral] = np.power(freqs[inspiral],-1./3.)
            dEdf_spectrum[merger] = np.power(freqs[merger],2./3.)/fMerge
            dEdf_spectrum[ringdown] = np.square(freqs[ringdown]/(1.+np.square((freqs[ringdown]-fRing)/(sigma/2.))))/(fMerge*fRing**(4./3.))


    # Normalization
    Mc = np.power(eta,3./5.)*Mtot*Msun
    amp = np.power(G*np.pi,2./3.)*np.power(Mc,5./3.)/3.
    return amp*dEdf_spectrum

def precompute_omega_weights(freqs, tmp_min=2., tmp_max=100., N=20000,chimax=None,inspiralOnly=False,PN=True):
    """
    Function to precompute the omega weights (adapted from https://ui.adsabs.harvard.edu/abs/2023arXiv231017625T/abstract)
    
    Parameters
    ----------
    freqs: array-like
        Array of frequencies for which to compute the energy spectrum radiated by the binaries.
    N: int
        Number of samples to generate, and to eventually perform the Monte-Carlo average on. Several
        values of this number have been tested, showing good convergence for the N=20000 case.
    tmp_min: float
        Minimum value used to draw the primary mass uniformly.
    tmp_max: float
        Maximum value used to draw the primary mass uniformly.
    chimax: float
        Add spins
        
    Returns
    -------
    m1s_drawn: array-like
        Array containing the uniformly drawn values of the primary mass
    m2s_drawn: array-like
        Array containing the uniformly drawn values of the secondary mass
    zs_drawn: array-like
        Array containing the uniformly drawn values of the redshifts
    p_m1_old: array-like
        Array containing the probability to draw the drawn m1 values, assuming a uniform distribution
    p_m2_old: array-like
        Array containing the probability to draw the drawn m2 values, assuming a uniform distribution
    p_z_old: array-like
        Array containing the probability to draw the drawn redshift values, assuming a uniform distribution
    dEdfs: array-like
        Array containing the energy density spectrum emitted by the binary with parameters given by the drawn parameters above.
    """

    m1s_drawn = np.random.uniform(tmp_min, tmp_max, size=N)
    c_m2s = np.random.uniform(size=int(N))
    m2s_drawn = tmp_min**(1.)+c_m2s*(m1s_drawn**(1.)-tmp_min**(1.))

    if chimax is not None:
        chi1,chi2 = np.random.uniform(0,chimax,size=N),np.random.uniform(0,chimax,size=N)
        delta_chi = (m1s_drawn-m2s_drawn)/(m1s_drawn+m2s_drawn)
        chi = 0.5*(1+delta_chi)*chi1+0.5*(1-delta_chi)*chi2 # From Ajith 2011 PRD, same reference above
        p_chi12 = np.power(1/chimax,2.) # Uniform prior for the two spins

    zs_drawn = np.random.uniform(0,10,size=N)

    if chimax is not None:
        dEdfs = np.array([dEdf(m1s_drawn[ii]+m2s_drawn[ii],freqs*(1+zs_drawn[ii]),eta=m2s_drawn[ii]/m1s_drawn[ii]/(1+m2s_drawn[ii]/m1s_drawn[ii])**2,
                              inspiralOnly=inspiralOnly,PN=PN,chi=chi[ii]) for ii in range(N)])
    else:
        dEdfs = np.array([dEdf(m1s_drawn[ii]+m2s_drawn[ii],freqs*(1+zs_drawn[ii]),eta=m2s_drawn[ii]/m1s_drawn[ii]/(1+m2s_drawn[ii]/m1s_drawn[ii])**2,
                              inspiralOnly=inspiralOnly,PN=PN) for ii in range(N)])

    p_m1_old = 1/(tmp_max-tmp_min)*np.ones(N)
    p_z_old = 1/(10-0)*np.ones(N)
    p_m2_old = 1/(m1s_drawn-tmp_min)

    look_up_Om0 = {'m1s_drawn':m1s_drawn,
              'm2s_drawn':m2s_drawn,
              'zs_drawn':zs_drawn,
              'p_m1_old':p_m1_old,
              'p_m2_old':p_m2_old,
              'p_z_old':p_z_old,
              'dEdfs':dEdfs}

    if chimax is not None:
        look_up_Om0['p_chi12']=p_chi12

    return look_up_Om0

# Define the log likelihood for the SGWB
def spectral_siren_vanilla_omega_gw(freqs,look_up_Om0,cbcrate):
    import time
    '''
    This function calculates the stochastic GW background as function of frequency. Note that this rate is not valid in modified gravity with friction terms

    Parameters
    ----------
    freqs: np.array
        Frequency at which to compute the power spectrum
    look_up_Om0: np.array
        Look up table computed with precompute_omega_weights
    cbcrate: class
        icarogw cbc rate for the spectral siren analysis

    Returns
    -------
    Omega_f: np.array
        The stochastic GW background
    '''
    c= astropy.constants.c.value
    G= astropy.constants.G.value
    km= 1.0e3
    Mpc= astropy.constants.kpc.value*1e3
    year= 365.*24*3600
    # Not valid in MG
    mw = cbcrate.mw
    rw = cbcrate.rw
    cw = cbcrate.cw
    R0 = cbcrate.R0
    H0 = cw.cosmology.little_h*100*km/Mpc  
    rhoC = 3.*np.power(H0*c,2.)/(8.*np.pi*G)*np.power(Mpc,3) 
    pmw=mw.pdf(look_up_Om0['m1s_drawn'],look_up_Om0['m2s_drawn'])
    prw=rw.evaluate(look_up_Om0['zs_drawn'])*R0
    p_rate_new = prw/cw.cosmology.astropy_cosmo.efunc(look_up_Om0['zs_drawn'])/(1.+look_up_Om0['zs_drawn']) 
    #formulae in the appendix
    w_i = p_rate_new*pmw/(look_up_Om0['p_z_old']*look_up_Om0['p_m1_old']*look_up_Om0['p_m2_old'])
    Omega_spectrum_new = np.einsum("if,i->if",look_up_Om0['dEdfs'],w_i)
    Omega_spectrum_new_avged = 1/rhoC/H0/1e9/year*freqs*np.mean(Omega_spectrum_new, axis=0)  
 
    Omega_f = Omega_spectrum_new_avged
    return Omega_f