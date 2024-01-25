# Use the python3 env to use icarogw2.0 package written below (and uncomment it)
import sys
import icarogw
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
from astropy.cosmology import FlatLambdaCDM, Planck15
import corner
from matplotlib import rc
from tqdm import tqdm as _tqdm

def chirp_mass_det(m1,m2,z):
    '''
    Fonction qui calcule la chirp mass d'un systeme binaire à partir de m1 et m2 (en Msol) 
    dans le detector frame. (facteur m_d = (1+z)*m_s)
    '''
    return (np.power(m1*m2,3/5)/np.power(m1+m2,1/5))*(1+z)

def chirp_mass(m1,m2):
    '''
    Fonction qui calcule la chirp mass d'un systeme binaire à partir de m1 et m2 (en Msol)
    '''
    return np.power(m1*m2,3/5)/np.power(m1+m2,1/5)

def mass_ratio(m1,m2):
    '''
    Computes the mass ratio (convention used is m1>m2 et q <1)
    '''
    return m2/m1

def f_GW(m1,m2,z):
    '''
    f_GW in detector frame in Hz
    '''
    m1 = m1*(1+z)
    m2 = m2*(1+z)
    M=m1+m2
    f_ISCO = (2.20*(1/M))*10**3
    to_ret = f_ISCO*2
    
    return to_ret

def z_to_dl(z):
    '''
    Convert a redshift value into a luminosity distance value.
    The Planck15 cosmology is used here. (but can be changed)
    Parameters:
    ----------
    z : float or array
        redshift to be converted
    '''
    cosmo = icarogw.cosmology.astropycosmology(14)
    cosmo.build_cosmology(Planck15)
    to_ret = cosmo.z2dl(z)
    return to_ret

def dl_to_z(z):
    '''
    Convert a dL value into a redshift value.
    The Planck15 cosmology is used here. (but can be changed)
    Parameters:
    ----------
    dL : float or array
        dL to be converted
    '''
    cosmo = icarogw.cosmology.astropycosmology(14)
    cosmo.build_cosmology(Planck15)
    to_ret = cosmo.dl2z(z)
    return to_ret

def dVc_dz(z):
    '''
    Compute the differential comoving volume as function of z
    '''
    cosmo = icarogw.cosmology.astropycosmology(14)
    cosmo.build_cosmology(Planck15)
    
    # differential of comov volume in Gpc3
    to_ret = cosmo.dVc_by_dzdOmega_at_z(z)*4*np.pi  
    
    return to_ret


def rvs_theta(Nsamp,a,b):
    '''
    Generate samples according to the CDF of theta in the range [a,b] with a<b
    By default, the full range is [0,1.4]
    
    '''
    cdf_theta = np.loadtxt("/Users/pierra/Desktop/These_ip2i/Cosmology_researches/icaroGW/icarogw_2/Sub_pop_BBH_ananalysis/data/Pw_three.dat")
    cdf_theta[:,1] =cdf_theta[:,1]/cdf_theta[:,1].max() 
    cdf_theta_func = interpolate.interp1d(cdf_theta[:,0],cdf_theta[:,1]) 
    inv_cdf_theta_func = interpolate.interp1d(cdf_theta[:,1],cdf_theta[:,0]) 
    
    
    unif_samples = np.random.random(Nsamp)*(cdf_theta_func(b)-cdf_theta_func(a)) + cdf_theta_func(a)
    theta = inv_cdf_theta_func(unif_samples)
    
    return theta

def dVc_dz_reweight(m1,m2,z):
    '''
    Perform an importance sampling with a weight (dVc/dz)*(1/1+z)
    Returns the reweighted quantities
    '''
    weight = dVc_dz(z)*1/(1+z)
    idx_resampling = np.random.choice(len(weight),replace=True,size=len(m1),p=weight/weight.sum())
    m1 = m1[idx_resampling]
    m2 = m2[idx_resampling]
    z = z[idx_resampling]
    
    return m1, m2, z


def snr_samples(m1,m2,z,numdet=3,rho_s=9,dL_s=1.5,Md_s=25,theta=None):
    '''
    Fonction that computes the SNR (approximation), using the formula showed in "Cosmo in the dark appendix"
    
    From m1s, m2s and z
    
    Parameters:
    ----------
    rho_s : float
         Reference value of the SNR 
    dL_s : float
        Reference value of the luminosity distance in [Gpc]
    Md_s : float
        Reference value for the chirp mass in [Msol]
    Md : float or array
        Chirp mass in the detector frame
    dL: float or array  
        Luminosity distance 
    theta : float or array
    '''

    Md = chirp_mass_det(m1,m2,z)
    dL = z_to_dl(z)
    
    if theta is None : 
        theta = rvs_theta(Nsamp=len(m1),a=0,b=1.4)
        rand_theta = np.random.choice(theta,len(m1))
        theta = rand_theta
    
    rho_true = rho_s*theta*np.power(Md/Md_s,5/6)*((dL_s*1000)/dL)
    rho_det_squarred = stats.ncx2.rvs(2*numdet,rho_true**2,size=len(rho_true))
    rho_det = np.power(rho_det_squarred,1/2)
    
    return rho_true , theta, rho_det    

def chirp_mass_noise(Md,rho_obs):
    '''
    Return the dected values of the chirp mass.
    Likelihood model based on Annexe B : https://arxiv.org/pdf/2103.14663.pdf
    '''
    Md_obs = np.random.randn(len(rho_obs))*(10**(-3))*Md*10/rho_obs + Md
    return Md_obs 

def mass_ratio_noise(q, rho_obs):
    '''
    Return the dected values of the mass ratio.
    Likelihood model based on Annexe B : https://arxiv.org/pdf/2103.14663.pdf
    '''
    q_obs = np.random.randn(len(rho_obs))*0.25*q*10/rho_obs + q
    return q_obs

def theta_noise(theta, rho_obs):
    '''
    Return the dected values of theta.
    Likelihood model based on Annexe B : https://arxiv.org/pdf/2103.14663.pdf
    '''
    
    theta_obs = np.random.randn(len(rho_obs))*0.3*10/rho_obs + theta
    return theta_obs

def noise(Md,q,theta,rho_obs):
    '''
    Use the 3 functions above to compute the 3 detected values for Md, q and theta
    '''
    
    Md_obs = chirp_mass_noise(Md,rho_obs)
    q_obs = mass_ratio_noise(q, rho_obs)
    theta_obs = theta_noise(theta, rho_obs)
    
    return Md_obs, q_obs, theta_obs


def snr_and_freq_cut(m1,m2,z,snr,snrthr=12,fgw_cut=15):
    '''
    Apply a cut in snr : snr > snrthr
    Apply a cut in frequency of the GWs in det frame : fGW > fgw_cut
    Returns the indices of each event that fits the two criterion
    '''
    
    freq_GW_astro = f_GW(m1,m2,z)
    indices = np.where((snr>=snrthr) & (freq_GW_astro>fgw_cut))[0]
    print('Ninj =',len(indices))
    return indices

def likelihood_evaluation(rhos,qs,Mds,thetas,rho_obs,q_obs,Md_obs,theta_obs,numdet=3):
    '''
    Compute the total likelihood of : the snr, the chirp mass, the mass ratio and theta
    All variables with an 's' at the end are the one we evaluate the likelihood at according to the '_obs' one.
    '''
    likelihood_tot = stats.ncx2.pdf(rho_obs**2., 2*numdet, np.power(rhos,2.))\
        *stats.norm.pdf(q_obs, qs, 0.25*qs*10/rho_obs)\
        *stats.norm.pdf(Md_obs, Mds, 10**(-3)*Mds*10/rho_obs)\
        *stats.norm.pdf(theta_obs, thetas, 0.3*10/rho_obs)
    
    return likelihood_tot

def generate_mass_inj(Nsamp):
    '''
    Generate Samples of m1 and m2 with PL+peak distribution
    '''
    mp = icarogw.wrappers.massprior_PowerLawPeak()
    mp.update(alpha=5.63,beta=15.26,mmin=30,mmax=250,delta_m=12.82,mu_g=100.,sigma_g=40.,lambda_peak=0.50)
    m1,m2 = mp.prior.sample(Nsamp)
    pdf = mp.prior.pdf(m1,m2)
    
    return m1, m2, pdf

def generate_dL_inj(alpha,beta,Nsamp):
    '''
    Generate Samples of dL propto a PL distribution (\prpto dL^2)
    The PL is defined in the interval [a,b]
    p(dL) \propto a*dL^(a-1) , with (a-1) =2
    '''
    dL_sample = stats.powerlaw.rvs(a=3,loc=alpha,scale=beta-alpha, size=Nsamp)
    
    pdf = stats.powerlaw.pdf(dL_sample, 3, loc=alpha, scale=beta-alpha)
    
    return dL_sample, pdf

def snr_samples_det(m1,m2,dL,numdet=3,rho_s=9,dL_s=1.5,Md_s=25,theta=None):
    '''
    Fonction that computes the SNR (approximation), using the formula showed in "Cosmo in the dark appendix"
    From m1d, m2d and dL
    
    Parameters:
    ----------
    rho_s : float
         Reference value of the SNR 
    dL_s : float
        Reference value of the luminosity distance in [Gpc]
    Md_s : float
        Reference value for the chirp mass in [Msol]
    Md : float or array
        Chirp mass in the detector frame
    dL: float or array  
        Luminosity distance 
    theta : float or array
    '''

    Md = chirp_mass(m1,m2)
   
    if theta is None : 
        theta = rvs_theta(Nsamp=len(m1),a=0,b=1.4)
        rand_theta = np.random.choice(theta,len(m1))
        theta = rand_theta
    
    rho_true = rho_s*theta*np.power(Md/Md_s,5/6)*((dL_s*1000)/dL)
    rho_det_squarred = stats.ncx2.rvs(2*numdet,rho_true**2,size=len(rho_true))
    rho_det = np.power(rho_det_squarred,1/2)
    
    return rho_true , theta, rho_det

def quick_data_preparation(m1_astro,m2_astro,zmerg_astro,numdet=3,rho_s=9,dL_s=1.5,Md_s=25,snrthr=12,fgw_cut=15,theta=None):
    '''
    Prepare the data that will be used as input in the function : PE_quick_generation_samples()
    
    Parameters
    ----------
    m1_astro : numpy array
        True distribution of m1 in source frame, flat in z
    m2_astro : numpy array
        True distribution of m2 in source frame, flat in z
    zmerg_astro :numpy array
        True distribution of the redshift merger, flat in z
    '''
    
    # Reweight by \frac{dV_{c}}{dz}\frac{1}{1+z}
    # N.B : It is optionnal, only useful if all astro distributions have been generated flat in z
    m1, m2, z = dVc_dz_reweight(m1_astro,m2_astro,zmerg_astro)
    
    # Compute associated : snr, Mc, q
    rho, theta, rho_obs = snr_samples(m1,m2,z,numdet=numdet,rho_s=rho_s,dL_s=dL_s,Md_s=Md_s,theta=theta)
    Md = chirp_mass_det(m1,m2,z)
    q = mass_ratio(m1,m2)
    
    # Compute the detected values : Md_obs, q_obs, theta_obs
    Md_obs, q_obs, theta_obs = noise(Md,q,theta,rho_obs)
    
    # Cut on the SNR and frequency of the GWs in det frame 
    idx_cut = snr_and_freq_cut(m1,m2,z,rho_obs,snrthr=snrthr,fgw_cut=fgw_cut)
    
    return m1,m2,z,theta,idx_cut,rho_obs,q_obs,Md_obs,theta_obs


def PE_quick_generation_samples(m1,m2,z,theta,idx,rho_obs,q_obs,Md_obs,theta_obs,Ninj=5,Nsamp=1000,numdet=3,rho_s=9,dL_s=1.5,Md_s=25,Ngen=int(1e7)):
    '''
    Generate PE given a list of noisy quantities 

    Parameters
    ----------
    m1, m2 : numpy array
        Primary masses m1 and m2 in the source frame
    z : numpy array
        Redshift of the source, corresponding to the masses
    theta : numpy array
        Parameter in [0,1.4] defined as https://arxiv.org/pdf/1405.7016.pdf
    idx : numpy array
        Indices of each signals that passe the snr&freq cut
    q_obs : numpy array
        Mass ratio observed
    Md_obs : numpy array
        Chirp mass in the detector frame observed
    theta_obs : numpy array
        Parameter in [0,1.4] observed
    Ninj : int
        Number of wanted signals 
    Nsamp: int
        Number of wanted posterior samples per signals
    numdet : integer
        number of detectors in the network, for the detection
    _s parameters : float
        Parameters that assess the sensitivity of the detector network we have
    _obs parameters :
        Notation used to describe the parameters that are drawn from the true ones, 
        according to the likelihood models of annex B : https://arxiv.org/pdf/2103.14663.pdf

    return
    ------
    dict1 : dictionnary
        Contains the PE samples for each event we generated and that are detected (i.e passes the snr&freq cut)
    dict2 : dictionnary
        Contains the true value of each parameters for each event. (a.k.a the true astrophysical distribution)
    idx : numpy array
        indices of the events for wich we generate PE samples

    '''
    
    idx = np.random.choice(idx,size=Ninj)
    dict1 = {}
    dict2 = {}
    
    for i in _tqdm(idx):
        
        
        # Draw trials solution for the masses and the distance : m1s, m2s dLs (uniformly)
        uncmass=1.1*0.7/(rho_obs[i]/8.)
        m1s = np.random.uniform(np.max([(1-uncmass)*m1[i],0.1]), (1+uncmass)*m1[i], size= Ngen)
        m2s = np.random.uniform(np.max([(1-uncmass)*m2[i],0.1]), (1+uncmass)*m2[i], size= Ngen)
        swap = np.where(m1s<m2s)[0]
        m1s[swap],m2s[swap]=m2s[swap],m1s[swap]
        
        uncdL=3*1.2/(rho_obs[i]/8.)
        dLs = np.random.uniform(np.max([(1-uncdL)*z_to_dl(z)[i],1.]),(1+uncdL)*z_to_dl(z)[i],size=Ngen)
        zs = dl_to_z(dLs)
              
        unctheta=15*4e-1/rho_obs[i]
        thetas = np.random.uniform(0,1.4,size=Ngen)
        
        qs = mass_ratio(m1s,m2s)
        Mds = chirp_mass_det(m1s,m2s,zs)
        
        rhos,_,_ =snr_samples(m1s,m2s,zs,numdet=numdet,rho_s=rho_s,dL_s=dL_s,Md_s=Md_s,theta=thetas)

        # Total likelihood estimate
        likelihood_tot=likelihood_evaluation(rhos,qs,Mds,thetas,rho_obs[i],q_obs[i],Md_obs[i],theta_obs[i],numdet=3)
        
        
        # Importance sampling
        proba = likelihood_tot/likelihood_tot.sum()
        idx_resampling = np.random.choice(Ngen,replace=True,size=Nsamp,p=proba)
        
        dict1[str(i)] = {'m1s_samp':m1s[idx_resampling],'m2s_samp':m2s[idx_resampling],
                         'zmerge_samples':zs[idx_resampling],'m1d_samples':m1s[idx_resampling]*(1+zs[idx_resampling]),
                        'm2d_samples':m2s[idx_resampling]*(1+zs[idx_resampling]),'Md_samples':Mds[idx_resampling],
                        'q_samples':qs[idx_resampling],'rho_samples':rhos[idx_resampling],'dL_samples':dLs[idx_resampling],
                         'theta_samples':thetas[idx_resampling]}
        
        dict2[str(i)] = {'m1s_samp':m1[i],'m2s_samp':m2[i],
                        'zmerge_samples':z[i],'m1d_samples':m1[i]*(1+z[i]),
                        'm2d_samples':m2[i]*(1+z[i]),'Md_samples':chirp_mass_det(m1[i],m2[i],z[i]),
                        'q_samples':mass_ratio(m1[i],m2[i]),'dL_samples':z_to_dl(z[i]),'theta_samples':theta[i]}
        
        
    return dict1, dict2, idx


def injection_set_generator(Ntot):
    '''
    This function generate the injection set to be given to IcaroGW to estimate the 
    selection effects. 
    Only a list of true parameters and the prior associated to each p(m1d,m2d)*p(dL) have to be stored.
    
    Parameters :
    -----------
    N_tot : Integer
        Total Number of injections we generate

    return
    ------
    true_param : dictionnary
        dictionnary that stores the true parameters of each injections that are detected
    Ndet : interger
        Number of detected injections
    '''
    true_param = {}
    
    # Generation of m1,m2 in det frame with a PL + peak
    m1d_inj, m2d_inj, pdf_m = generate_mass_inj(Nsamp=Ntot)
    # Generation of dL with a PL ( \propto dL^2)
    alpha,beta = 44, 26039 # Mpc
    dL_inj, pdf_dL = generate_dL_inj(alpha=alpha,beta=beta,Nsamp=Ntot)
    # Store the associated total prior
    prior_inj = pdf_m*pdf_dL
    
    # Compute the associated quantites to derive the true SNR
    Mc_inj = chirp_mass(m1d_inj,m2d_inj)
    q_inj = mass_ratio(m1d_inj,m2d_inj)
    snr_inj,_,_ = snr_samples_det(m1d_inj,m2d_inj,dL_inj,numdet=3,rho_s=9,dL_s=1.5,Md_s=25,theta=None)
    
    # Compute the true parameters to be able to cut over the freq and the snr
    z_inj = dl_to_z(dL_inj)
    m1s_inj, m2s_inj = m1d_inj/(1+z_inj), m2d_inj/(1+z_inj)
    
    # Cut on freq and SNR
    idx_detected_inj = snr_and_freq_cut(m1s_inj,m2s_inj,z_inj,snr_inj,snrthr=12,fgw_cut=15)
    Ndet = len(idx_detected_inj)
    true_param = {'m1s':m1s_inj[idx_detected_inj],'m2s':m2s_inj[idx_detected_inj],'z':z_inj[idx_detected_inj],
               'snr':snr_inj[idx_detected_inj],'m1d':m1d_inj[idx_detected_inj],'m2d':m2d_inj[idx_detected_inj],
                 'dL':dL_inj[idx_detected_inj],'prior':prior_inj[idx_detected_inj]}
    
    return true_param, Ntot, Ndet