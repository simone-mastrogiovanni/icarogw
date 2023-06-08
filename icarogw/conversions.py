from .jax_pal import *
import healpy as hp
from tqdm import tqdm

def cartestianspins2chis(s1x,s1y,s1z,s2x,s2y,s2z,q):
    '''
    Returns chi_1, chi_2, cosines with orbital angular momentum, chi_eff and chi_p 
    from cartesian spin components (z is on the orbital angular momentum).
    
    Parameters
    ----------
    s1x,s1y,s1z: jnp.array
        Spin cartesian components of first object
    s2x,s2y,s2z: jnp.array
        Spin cartesian components of secondary object
    q: jnp.array
        mass ratio in (0,1]
    '''
    
    chi_1 = jnp.sqrt(s1x**2.+s1y**2.+s1z**2.)
    chi_2 = jnp.sqrt(s2x**2.+s2y**2.+s2z**2.)
    cos_t_1, cos_t_2 = s1z/chi_1 , s2z/chi_2
    chi_eff = chi_eff_from_spins(chi_1, chi_2, cos_t_1, cos_t_2, q)
    chi_p = chi_p_from_spins(chi_1, chi_2, cos_t_1, cos_t_2, q)
    
    return chi_1, chi_2, cos_t_1, cos_t_2, chi_eff, chi_p

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

    return spence(1.-z+0j)

def chi_effective_prior_from_aligned_spins(q,aMax,xs):
    '''
    Function from https://github.com/tcallister/effective-spin-priors. Credit: T. Callister
    Derivation of equations can be found in arxiv:2104.09508
    
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, aligned component spin priors.
    
    Parameters
    ----------
    q: onp.array
        Mass ratio value (according to the convention q<1)
    aMax: float
        Maximum allowed dimensionless component spin magnitude
    xs: onp.array
        Chi_effective value or values at which we wish to compute prior
  
    Returns
    -------
    Array of prior values
    '''

    # Ensure that `xs` is an array and take Absolute bolometric value
    origin_shape=xs.shape
    xs = onp.reshape(xs,-1)

    # Set up various piecewise cases
    pdfs = onp.zeros(xs.size)
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
    q: onp.array
        Mass ratio value (according to the convention q<1)
    aMax: float
        Maximum allowed dimensionless component spin magnitude
    xs: onp.array
        Chi_effective value or values at which we wish to compute prior
    
    Returns
    -------
    Array of prior values
    '''

    # Ensure that `xs` is an array and take Absolute bolometric value
    origin_shape=xs.shape
    xs = onp.reshape(onp.abs(xs),-1)

    # Set up various piecewise cases
    pdfs = onp.ones(xs.size,dtype=complex)*(-1.)
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

    pdfs[caseZ] = (1.+q_Z)/(2.*aMax)*(2.-onp.log(q_Z))

    pdfs[caseA] = (1.+q_A)/(4.*q_A*aMax**2)*(
                    q_A*aMax*(4.+2.*onp.log(aMax) - onp.log(q_A**2*aMax**2 - (1.+q_A)**2*x_A**2))
                    - 2.*(1.+q_A)*x_A*onp.arctanh((1.+q_A)*x_A/(q_A*aMax))
                    + (1.+q_A)*x_A*(Di(-q_A*aMax/((1.+q_A)*x_A)) - Di(q_A*aMax/((1.+q_A)*x_A)))
                    )

    pdfs[caseB] = (1.+q_B)/(4.*q_B*aMax**2)*(
                    4.*q_B*aMax
                    + 2.*q_B*aMax*onp.log(aMax)
                    - 2.*(1.+q_B)*x_B*onp.arctanh(q_B*aMax/((1.+q_B)*x_B))
                    - q_B*aMax*onp.log((1.+q_B)**2*x_B**2 - q_B**2*aMax**2)
                    + (1.+q_B)*x_B*(Di(-q_B*aMax/((1.+q_B)*x_B)) - Di(q_B*aMax/((1.+q_B)*x_B)))
                    )

    pdfs[caseC] = (1.+q_C)/(4.*q_C*aMax**2)*(
                    2.*(1.+q_C)*(aMax-x_C)
                    - (1.+q_C)*x_C*onp.log(aMax)**2.
                    + (aMax + (1.+q_C)*x_C*onp.log((1.+q_C)*x_C))*onp.log(q_C*aMax/(aMax-(1.+q_C)*x_C))
                    - (1.+q_C)*x_C*onp.log(aMax)*(2. + onp.log(q_C) - onp.log(aMax-(1.+q_C)*x_C))
                    + q_C*aMax*onp.log(aMax/(q_C*aMax-(1.+q_C)*x_C))
                    + (1.+q_C)*x_C*onp.log((aMax-(1.+q_C)*x_C)*(q_C*aMax-(1.+q_C)*x_C)/q_C)
                    + (1.+q_C)*x_C*(Di(1.-aMax/((1.+q_C)*x_C)) - Di(q_C*aMax/((1.+q_C)*x_C)))
                    )

    pdfs[caseD] = (1.+q_D)/(4.*q_D*aMax**2)*(
                    -x_D*onp.log(aMax)**2
                    + 2.*(1.+q_D)*(aMax-x_D)
                    + q_D*aMax*onp.log(aMax/((1.+q_D)*x_D-q_D*aMax))
                    + aMax*onp.log(q_D*aMax/(aMax-(1.+q_D)*x_D))
                    - x_D*onp.log(aMax)*(2.*(1.+q_D) - onp.log((1.+q_D)*x_D) - q_D*onp.log((1.+q_D)*x_D/aMax))
                    + (1.+q_D)*x_D*onp.log((-q_D*aMax+(1.+q_D)*x_D)*(aMax-(1.+q_D)*x_D)/q_D)
                    + (1.+q_D)*x_D*onp.log(aMax/((1.+q_D)*x_D))*onp.log((aMax-(1.+q_D)*x_D)/q_D)
                    + (1.+q_D)*x_D*(Di(1.-aMax/((1.+q_D)*x_D)) - Di(q_D*aMax/((1.+q_D)*x_D)))
                    )

    pdfs[caseE] = (1.+q_E)/(4.*q_E*aMax**2)*(
                    2.*(1.+q_E)*(aMax-x_E)
                    - (1.+q_E)*x_E*onp.log(aMax)**2
                    + onp.log(aMax)*(
                        aMax
                        -2.*(1.+q_E)*x_E
                        -(1.+q_E)*x_E*onp.log(q_E/((1.+q_E)*x_E-aMax))
                        )
                    - aMax*onp.log(((1.+q_E)*x_E-aMax)/q_E)
                    + (1.+q_E)*x_E*onp.log(((1.+q_E)*x_E-aMax)*((1.+q_E)*x_E-q_E*aMax)/q_E)
                    + (1.+q_E)*x_E*onp.log((1.+q_E)*x_E)*onp.log(q_E*aMax/((1.+q_E)*x_E-aMax))
                    - q_E*aMax*onp.log(((1.+q_E)*x_E-q_E*aMax)/aMax)
                    + (1.+q_E)*x_E*(Di(1.-aMax/((1.+q_E)*x_E)) - Di(q_E*aMax/((1.+q_E)*x_E)))
                    )

    pdfs[caseF] = 0.

    # Deal with spins on the boundary between cases
    if onp.any(pdfs==-1):
        boundary = (pdfs==-1)
        pdfs[boundary] = 0.5*(chi_effective_prior_from_isotropic_spins(q[boundary],aMax,xs[boundary]+1e-6)\
                        + chi_effective_prior_from_isotropic_spins(q[boundary],aMax,xs[boundary]-1e-6))

    return onp.real(pdfs).reshape(origin_shape)

def chi_p_prior_from_isotropic_spins(q,aMax,xs):
    '''
    Function from https://github.com/tcallister/effective-spin-priors. Credit: T. Callister
    Derivation of equations can be found in arxiv:2104.09508
    
    Function defining the conditional priors p(chi_p|q) corresponding to
    uniform, isotropic component spin priors.

    Parameters
    ----------
    q: onp.array
        Mass ratio value (according to the convention q<1)
    aMax: float
        Maximum allowed dimensionless component spin magnitude
    xs: onp.arry
        Chi_p value or values at which we wish to compute prior
    
    Returns:
    --------
    Array of prior values
    '''

    # Ensure that `xs` is an array and take Absolute bolometric value
    origin_shape=xs.shape
    xs = onp.reshape(xs,-1)

    # Set up various piecewise cases
    pdfs = onp.zeros(xs.size)
    caseA = xs<q*aMax*(3.+4.*q)/(4.+3.*q)
    caseB = (xs>=q*aMax*(3.+4.*q)/(4.+3.*q))*(xs<aMax)

    # Select relevant effective spins
    x_A, q_A = xs[caseA], q[caseA]
    x_B, q_B = xs[caseB], q[caseB]

    pdfs[caseA] = (1./(aMax**2*q_A))*((4.+3.*q_A)/(3.+4.*q_A))*(
                    onp.arccos((4.+3.*q_A)*x_A/((3.+4.*q_A)*q*aMax))*(
                        aMax
                        - onp.sqrt(aMax**2-x_A**2)
                        + x_A*onp.arccos(x_A/aMax)
                        )
                    + onp.arccos(x_A/aMax)*(
                        aMax*q_A*(3.+4.*q_A)/(4.+3.*q_A)
                        - onp.sqrt(aMax**2*q_A**2*((3.+4.*q_A)/(4.+3.*q_A))**2 - x_A**2)
                        + x_A*onp.arccos((4.+3.*q_A)*x_A/((3.+4.*q_A)*aMax*q_A))
                        )
                    )

    pdfs[caseB] = (1./aMax)*onp.arccos(x_B/aMax)

    return pdfs.reshape(origin_shape)

def joint_prior_from_isotropic_spins(q,aMax,xeffs,xps,ndraws=10000,bw_method='scott'):
    '''
    Function from https://github.com/tcallister/effective-spin-priors. Credit: T. Callister
    Derivation of equations can be found in arxiv:2104.09508
    
    Function to calculate the conditional priors p(jnp|xeff,q) on a set of {jnp,xeff,q} posterior samples.
    
    Parameters
    ----------
    q: onp.array
        Mass ratio
    aMax: float
        Maximimum spin magnitude considered
    xeffs: onp.array
        Effective inspiral spin samples
    xps: onp.array
        Effective precessing spin values
    ndraws: int
        Number of draws from the component spin priors used in numerically building interpolant
    
    Returns
    -------
    
    Array of priors on jnp, conditioned on given effective inspiral spins and mass ratios
    '''

    # Convert to arrays for safety
    origin_shape=xeffs.shape
    xeffs = onp.reshape(xeffs,-1)
    xps = onp.reshape(xps,-1)

    # Compute marginal prior on xeff, conditional prior on jnp, and multiply to get joint prior!
    p_chi_eff = chi_effective_prior_from_isotropic_spins(q,aMax,xeffs)
    p_chi_p_given_chi_eff = onp.zeros(len(p_chi_eff))
    
    for i in tqdm(range(len(p_chi_eff)),desc='Calculating p(chi_p|chi_eff,q)'):
        p_chi_p_given_chi_eff[i] = chi_p_prior_given_chi_eff_q(q[i],aMax,xeffs[i],xps[i],ndraws,bw_method)
    joint_p_chi_p_chi_eff = p_chi_eff*p_chi_p_given_chi_eff

    return joint_p_chi_p_chi_eff.reshape(origin_shape)

def chi_p_prior_given_chi_eff_q(q,aMax,xeff,jnp,ndraws=10000,bw_method='scott'):
    '''
    Function from https://github.com/tcallister/effective-spin-priors. Credit: T. Callister
    Derivation of equations can be found in arxiv:2104.09508
    
    Function to calculate the conditional prior p(jnp|xeff,q) on a single {jnp,xeff,q} posterior sample.
    Called by `joint_prior_from_isotropic_spins`.
    
    Parameters
    ----------
    q: onp.array
        Single posterior mass ratio sample
    aMax: float
        Maximimum spin magnitude considered
    xeff: onp.array
        Single effective inspiral spin sample
    jnp: onp.array
        Single effective precessing spin value
    ndraws: int
        Number of draws from the component spin priors used in numerically building interpolant
    
    Returns
    -------
    Prior on jnp, conditioned on given effective inspiral spin and mass ratio
    '''

    # Draw random spin magnitudes.
    # Note that, given a fixed chi_eff, a1 can be no larger than (1+q)*chi_eff,
    # and a2 can be no larger than (1+q)*chi_eff/q
    a1 = onp.random.random(ndraws)*aMax
    a2 = onp.random.random(ndraws)*aMax

    # Draw random tilts for spin 2
    cost2 = 2.*onp.random.random(ndraws)-1.

    # Finally, given our conditional value for chi_eff, we can solve for cost1
    # Note, though, that we still must require that the implied value of cost1 be *physical*
    cost1 = (xeff*(1.+q) - q*a2*cost2)/a1

    # While any cost1 values remain unphysical, redraw a1, a2, and cost2, and recompute
    # Repeat as necessary
    while onp.any(cost1<-1) or onp.any(cost1>1):
        to_replace = onp.where((cost1<-1) | (cost1>1))[0]
        a1[to_replace] = onp.random.random(to_replace.size)*aMax
        a2[to_replace] = onp.random.random(to_replace.size)*aMax
        cost2[to_replace] = 2.*onp.random.random(to_replace.size)-1.
        cost1 = (xeff*(1.+q) - q*a2*cost2)/a1

    # Compute precessing spins and corresponding weights, build KDE
    # See `Joint-ChiEff-ChiP-Prior.ipynb` for a discussion of these weights
    Xp_draws = chi_p_from_spins(a1,a2,cost1,cost2,q)
    jacobian_weights = (1.+q)/a1
    prior_kde = gaussian_kde(Xp_draws,weights=jacobian_weights,bw_method=bw_method)

    # Compute maximum chi_p
    if (1.+q)*onp.abs(xeff)/q<aMax:
        max_Xp = aMax
    else:
        max_Xp = onp.sqrt(aMax**2 - ((1.+q)*onp.abs(xeff)-q)**2.)

    # Set up a grid slightly inside (0,max chi_p) and evaluate KDE
    reference_grid = onp.linspace(0.05*max_Xp,0.95*max_Xp,50)
    reference_vals = prior_kde(reference_grid)

    # Manually prepend/append zeros at the boundaries
    reference_grid = onp.concatenate([[0],reference_grid,[max_Xp]])
    reference_vals = onp.concatenate([[0],reference_vals,[0]])
    norm_constant = onp.trapz(reference_vals,reference_grid)

    # Interpolate!
    p_chi_p = onp.interp(jnp,reference_grid,reference_vals/norm_constant)
    return p_chi_p

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
    
    sin1 = onp.sqrt(1.-cos1**2)
    sin2 = onp.sqrt(1.-cos2**2)

    to_ret = onp.maximum(chi1*sin1, ((4*q+3)/(3*q+4))*q*chi2*sin2)
    
    return to_ret

def radec2skymap(ra,dec,nside):
    '''
    Converts RA and DEC samples to a skymap with normalized probability, i.e. integral of skymap in dArea =1
    
    Parameters
    ----------
    ra, dec: onp.array
        arrays with RA and DEC in radians
    nside: int
        nside for healpy
    
    Returns
    -------
    counts_maps: onp.array
        Healpy array with skymap
    dOmega_sterad: float
        Area in steradians of the sky cell
    '''

    npixels=hp.nside2npix(nside)
    dOmega_sterad=4*onp.pi/npixels
    dOmega_deg2=onp.power(180/jnp.pi,2.)*dOmega_sterad
    indices = radec2indeces(ra,dec,nside)
    counts_map = onp.zeros(npixels)
    for indx in range(npixels):
        ind=onp.where(indices==indx)[0]
        counts_map[indx]=len(ind)
        if ind.size==0:
            continue
    counts_map/=(len(ra)*dOmega_sterad)
    return counts_map, dOmega_sterad


def L2M(L):
    '''
    Converts Luminosity in Watt to Absolute bolometric magnitude
    
    Parameters
    ----------
    L: jnp.array
        Luminosity in Watt
    
    Returns
    -------
    M: jnp.array
        Absolute bolometric magnitude
    '''
    # From Resolution B2 proposed by IAU, see e.g. Pag. 2, Eq. 2 of https://www.iau.org/static/resolutions/IAU2015_English.pdf
    return -2.5*jnp.log10(L)+71.197425

def M2L(M):
    '''
    Converts Absolute bolometric magnitude to Luminosity in Watt 
    
    Parameters
    ----------
    M: jnp.array
        Absolute bolometric magnitude
    
    Returns
    -------
    L: jnp.array
        Luminosity in Watt
    '''
    # From Pag. 2, Eq. 1-2 of https://www.iau.org/static/resolutions/IAU2015_English.pdf
    return 3.0128e28*jnp.power(10.,-0.4*M)

def radec2indeces(ra,dec,nside):
    '''
    Converts RA and DEC to healpy indeces
    
    Parameters
    ----------
    ra, dec: onp.array
        arrays with RA and DEC in radians
    nside: int
        nside for healpy
    
    Returns
    -------
    healpy indeces as numpy array
    '''
    theta = jnp.pi/2.0 - dec
    phi = ra
    return hp.ang2pix(nside, theta, phi)

def indices2radec(indices,nside):
    '''
    Converts healpy indeces to RA DEC
    
    Parameters
    ----------
    indices: onp.array
        Healpy indices
    nside: int
        nside for healpy
    
    Returns
    -------
    ra, dec: onp.array
        arrays with RA and DEC in radians
    '''
    
    theta,phi= hp.pix2ang(nside,indices)
    return phi, jnp.pi/2.0-theta

def M2m(M,dl,kcorr):
    '''
    Converts Absolute bolometric magnitude to apparent magnitude
    
    Parameters
    ----------
    M: jnp.array
        Absolute bolometric magnitude
    dl: jnp.array
        Luminosity distance in Mpc
    kcorr: jnp.array
        K-correction, we suggest to use the kcorr class
    
    Returns
    -------
    m: jnp.array
        Apparent magnitude
    '''
    # Note that the distance is Mpc here. See Eq. 2 of https://arxiv.org/abs/astro-ph/0210394
    dist_modulus=5*jnp.log10(dl)+25.
    return M+dist_modulus+kcorr

def m2M(m,dl,kcorr):
    '''
    Converts apparent magnitude to Absolute bolometric magnitude
    
    Parameters
    ----------
    m: jnp.array
        apparebt magnitude
    dl: jnp.array
        Luminosity distance in Mpc
    kcorr: jnp.array
        K-correction, we suggest to use the kcorr class
    
    Returns
    -------
    M: jnp.array
        Absolute bolometric magnitude
    '''
    # Note that the distance is Mpc here. See Eq. 2 of https://arxiv.org/abs/astro-ph/0210394
    dist_modulus=5*jnp.log10(dl)+25.
    return m-dist_modulus-kcorr

def source2detector(mass1_source,mass2_source,z,cosmology):
    '''
    Converts from Source frame to detector frame quantities
    
    Parameters
    ----------
    mass1_source: jnp.array
        Source mass of the primary
    mass2_source: jnp.array
        Source mass of the secondary
    z: jnp.array
        Redshift of the object
    cosmology: class
        Cosmology class from the cosmology module
    
    Returns
    -------
    Detector frame masses and luminosity distance in Mpc
    '''
    return mass1_source*(1+z),mass2_source*(1+z),cosmology.z2dl(z)

def detector2source(mass1,mass2,dl,cosmology):
    '''
    Converts from Source frame to detector frame quantities
    
    Parameters
    ----------
    mass1: jnp.array
        Detector mass of the primary
    mass2: jnp.array
        Detector mass of the secondary
    dl: jnp.array
        Luminosity distnce in Mpc
    cosmology: class
        Cosmology class from the cosmology module
    
    Returns
    -------
    Source frame masses and redshift
    '''
    
    z=cosmology.dl2z(dl)
    return mass1/(1+z),mass2/(1+z),z


def detector2source_jacobian(z, cosmology):
    '''
    Calculates the detector frame to source frame Jacobian d_det/d_sour

    Parameters
    ----------
    z: jnp. arrays
        Redshift
    cosmo:  class from the cosmology module
        Cosmology class from the cosmology module
    '''
    return jnp.abs(jnp.power(1+z,2.)*cosmology.ddl_by_dz_at_z(z))
    
def source2detector_jacobian(z, cosmology):
    '''
    Calculates the detector frame to source frame Jacobian d_sour/d_det

    Parameters
    ----------
    z: jnp. arrays
        Redshift
    cosmo:  class from the cosmology module
        Cosmology class from the cosmology module
    '''
    return 1./detector2source_jacobian(z,cosmology)
    
