import icarogw
import h5py
import zarr
import numpy as np
from tqdm import tqdm
from astropy.cosmology import Planck15
import matplotlib.pyplot as plt
from mhealpy import HealpixBase, HealpixMap

cosmo_ref = icarogw.cosmology.astropycosmology(zmax=10.)
cosmo_ref.build_cosmology(Planck15)

outfolder = '../glade_pixelated'
nside = 64
grouping = 'K-band'
subgrouping = 'eps-1'
apparent_magnitude_flag = 'm_K'
nside_mthr = 32
epsilon = 1.

with h5py.File('../glade+.hdf5','r') as glade:
    icarogw.catalog.create_pixelated_catalogs(outfolder,nside,{key:glade[key] for key in ['ra','dec','m_K','z','sigmaz','m_W1']}
                                             ,batch=100000,nest=False)
    
icarogw.catalog.clear_empty_pixelated_files(outfolder,nside)

filled_pixels = np.genfromtxt('../glade_pixelated/filled_pixels.txt').astype(int)
for pix in tqdm(filled_pixels):
    icarogw.catalog.remove_nans_pixelated_files(outfolder,pix,
                                                ['ra','dec','m_K','z','sigmaz'],
                                                grouping)

for pix in tqdm(filled_pixels):
    icarogw.catalog.calculate_mthr_pixelated_files(outfolder,
                                   pix,
                                   apparent_magnitude_flag,grouping,nside_mthr,
                                   mthr_percentile=50)

for pix in tqdm(filled_pixels):
    icarogw.catalog.get_redshift_grid_for_files(outfolder,pix,grouping,cosmo_ref,
                               Nintegration=10,Numsigma=3,zcut=0.5)

icarogw.catalog.initialize_icarogw_catalog(outfolder,'icarogw_gladep.hdf5',grouping)

dd = h5py.File('icarogw_gladep.hdf5','r')
z_grid = dd['K-band/z_grid'][:]
dd.close()

filled_pixels = np.genfromtxt('../glade_pixelated/filled_pixels.txt').astype(int)
for pix in tqdm(filled_pixels):

    icarogw.catalog.calculate_interpolant_files(outfolder,z_grid,pix,grouping,subgrouping,
                                'K',cosmo_ref,epsilon,ptype='gaussian')