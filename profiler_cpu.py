import os

fp=open('config.py','w')
fp.write('CUPY=False')
fp.close()


import icarogwCAT
from icarogwCAT.cupy_pal import *
import matplotlib.pyplot as plt
import numpy as np
import corner
import h5py
from astropy.cosmology import FlatLambdaCDM
import healpy as hp
import copy as cp
import time
from tqdm import tqdm

os.remove('config.py')

from astropy.cosmology import Planck15
cosmo_prova=icarogwCAT.cosmology.FLRWcosmology(1.)
cosmo_prova.build_cosmology(Planck15)

nside=64

GW170817=h5py.File('/home/smastro/Desktop/PE/GWTC-1/GW170817.hdf5')
ppd={'mass_1':GW170817['IMRPhenomPv2NRT_highSpin_posterior']['m1_detector_frame_Msun'],
    'mass_2':GW170817['IMRPhenomPv2NRT_highSpin_posterior']['m2_detector_frame_Msun'],
    'luminosity_distance':GW170817['IMRPhenomPv2NRT_highSpin_posterior']['luminosity_distance_Mpc'],
    'right_ascension':GW170817['IMRPhenomPv2NRT_highSpin_posterior']['right_ascension'],
    'declination':GW170817['IMRPhenomPv2NRT_highSpin_posterior']['declination']}

cosmo_ref=icarogwCAT.cosmology.FLRWcosmology(zmax=5.)
cosmo_ref.build_cosmology(FlatLambdaCDM(H0=67.7,Om0=0.308))

cat=icarogwCAT.catalog.galaxy_catalog('../glade+.hdf5','K',nside,mthr_percentile=50,Lfuncmod=1.)
cat.remove_mthr_galaxies()
cat.calc_dN_by_dzdOmega_interpolant(cosmo_ref,Nres=1,zcut=0.1,ptype='gaussian')



mw=icarogwCAT.wrappers.massprior_PowerLaw()
mw.update(alpha=0.,beta=0.,mmin=1,mmax=3.)
mw.prior

rw=icarogwCAT.wrappers.rateevolution_Madau()
rw.update(gamma=0.,kappa=6.0,zp=2.4)

data=h5py.File('../endo3_bnspop-LIGO-T2100113-v12.hdf5')
cosmo_ref=icarogwCAT.cosmology.FLRWcosmology(10)
cosmo_ref.build_cosmology(Planck15)

ifarmax=np.vstack([data['injections'][key] for key in ['ifar_cwb', 'ifar_gstlal', 'ifar_mbta', 'ifar_pycbc_bbh', 'ifar_pycbc_hyperbank']])
ifarmax=np.max(ifarmax,axis=0)
time_O3 = (28519200/86400)/365 # Time of observation for O3
   

prior=np2cp(data['injections/mass1_source_mass2_source_sampling_pdf'][()]*data['injections/redshift_sampling_pdf'][()]/(np.pi*4))
prior*=icarogwCAT.conversions.source2detector_jacobian(np2cp(data['injections/redshift'][()]),cosmo_ref)

injections_dict={'mass_1':data['injections/mass1'][()],'mass_2':data['injections/mass2'][()],
                'luminosity_distance':data['injections/distance'][()],
                'right_ascension':data['injections/right_ascension'][()],'declination':data['injections/declination'][()]}
inj=icarogwCAT.injections.detector_injections_v1(injections_dict,prior=prior,ntotal=data.attrs['total_generated'],Tobs=time_O3)
inj.update_cut(ifarmax>=4)
inj.pixelize(nside=64)
inj.cupyfy()



timing=np.zeros([500,5])
Nevv=np.array([1, 10, 100, 1000, 10000])

for j,Nev in enumerate(Nevv):
    posterior_dict={}
    for i in range(Nev):
        posterior_dict['GW'+str(i)]=icarogwCAT.posterior_samples.posterior_samples(ppd,
                                                                         prior=np.power(GW170817['IMRPhenomPv2NRT_highSpin_posterior']['luminosity_distance_Mpc'],2.))
    
    
    likelihood=icarogwCAT.likelihood.hierarchical_likelihood_galaxies(posterior_dict,inj,cat,'FlatLambdaCDM','PowerLaw','Madau',
                                                                      zmax=2.,nparallel=2048,scale_free=True)
    for i in tqdm(range(500)):
        likelihood.parameters={'alpha': 0.,
     'beta': 0.,
     'mmin': 1.,
     'mmax': 2.,
     'gamma': 2.5,
     'kappa': 6.,
     'zp': 2.5,
     'H0': 70.,
     'Om0': 0.308}
        start=time.time()
        _=likelihood.log_likelihood()
        end=time.time()
        timing[i,j]=end-start
    
np.savetxt('timing_cpu.txt',cp2np(timing))
    
    