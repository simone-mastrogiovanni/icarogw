from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn, is_there_cupy
from .conversions import radec2indeces, indices2radec, M2m, m2M
from .cosmology import galaxy_MF, log_powerlaw_absM_rate

import healpy as hp
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import os
import shutil
from mhealpy import HealpixBase, HealpixMap

LOWERL=np.nan_to_num(-np.inf)


def create_pixelated_catalogs(outfolder,nside,groups_dict,batch=100000,nest=False):
    '''
    Divide a galaxy catlalog file into pixelated files

    Parameters
    ----------
    outfolder: str
        Path where to save the pixels files
    nside: int
        Nside for the healpy pixelization
    groups_dict: int
        A dictionary containing the h5py datasets to save. The dictionary should have at least
        'ra' [rad], 'dec' [rad], 'z' and 'sigmaz and some apparent magnitude
    batch: int
        How many galaxies to process
    nest: bool
        Nest flag for healpy
    '''
    list_of_keys = list(groups_dict.keys())
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
        for i in tqdm(range(hp.nside2npix(nside)),desc='Creating the file with pixels fields'):    
            cat = h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(i)),'w-')
            cat.create_group('catalog')
            cat.attrs['nside']=nside
            cat.attrs['nest']=nest
            cat.attrs['dOmega_sterad']=hp.nside2pixarea(nside,degrees=False)
            cat.attrs['dOmega_deg2']=hp.nside2pixarea(nside,degrees=True)
            cat.attrs['Ntotal_galaxies_original']=0
            for key in list_of_keys:
                cat['catalog'].create_dataset(key, data=np.array([]), compression="gzip", chunks=True, maxshape=(None,))
            cat.close()

        istart = 0
        np.savetxt(os.path.join(outfolder,'checkpoint_creation.txt'),np.array([istart]),fmt='%d')    
        
    else:
        istart = np.genfromtxt(os.path.join(outfolder,'checkpoint_creation.txt')).astype(int)
    
    Ntotal = len(groups_dict['ra'])  
    pbar = tqdm(total=Ntotal-istart)

    while istart < Ntotal:
        idx = radec2indeces(groups_dict['ra'][istart:istart+batch],groups_dict['dec'][istart:istart+batch],nside,nest=nest)
        u, indices = np.unique(idx, return_inverse=True) # u array of unique indices
        for ipix, pixel in enumerate(u):
            with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pixel)),'r+') as pp:
                pix = pp['catalog']
                galaxies_id = np.where(indices==ipix)[0] # They are all the galaxies staying in this pixel
                pp.attrs['Ntotal_galaxies_original']+=len(galaxies_id)
                for key in list_of_keys:
                    pix[key].resize((pix[key].shape[0] + len(galaxies_id)), axis = 0)
                    pix[key][-len(galaxies_id):] = groups_dict[key][istart:istart+batch][galaxies_id]
        istart+=batch
        pbar.update(batch)
        np.savetxt(os.path.join(outfolder,'checkpoint_creation.txt'),np.array([istart]),fmt='%d')
    pbar.close()

def clear_empty_pixelated_files(outfolder,nside):
    '''
    This function deletes the files with empty pixels and also saves a 
    file in outfolder/filled_pixels.txt with the pixel labels of the filled 
    pixels

    Parameters
    ----------
    outfolder: str
        Where the pixel files are saved
    nside: int
        nside used for the analysis
    '''
    filled_pixels = []
    for pix in tqdm(range(hp.nside2npix(nside)),desc='Checking pixel files'):
        if not os.path.isfile(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pix))):
            continue
        with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pix)),'r') as subcat:
            if subcat.attrs['Ntotal_galaxies_original']!=0:
                filled_pixels.append(pix)
            else:
                os.remove(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pix)))
    np.savetxt(os.path.join(outfolder,'filled_pixels.txt'),np.array(filled_pixels),fmt='%d')

def remove_nans_pixelated_files(outfolder,pixel,fields_to_take,grouping):
    '''
    The function creates a group in the pixelated files and record what galaxies have 
    NaNs.

    Parameters
    ----------
    outfolder: str
        Where are the pixelated files
    pixel: int
        The pixel label you want to read
    fields_to_take: list
        List of strings of the field in the group you want to take
    grouping: str
        How the new group should be called
    '''
    with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pixel)),'r+') as cat:
        gg=cat.require_group(grouping)
        if 'NaNs_removed' not in list(gg.attrs.keys()):
            gg.attrs['NaNs_removed'] = False

        if not gg.attrs['NaNs_removed']:
            bigbool = np.isfinite(np.vstack([cat['catalog'][key][:] for key in fields_to_take]))
            tokeep = np.all(bigbool,axis=0)
            gg.create_dataset('not_NaN_indices',data=tokeep)
            gg.attrs['NaNs_removed'] = True
        
def calculate_mthr_pixelated_files(outfolder,
                                   pixel,
                                   apparent_magnitude_flag,grouping,nside_mthr,
                                   mthr_percentile=50):
    '''
    The function calculates the apparent magnitude threshold for each pixelated file

    Parameters
    ----------
    outfolder: str
        Where are the pixelated files
    pixel: int
        The pixel label you want to read
    apparent_magnitude_flag: str
        The flag of the apparent magnitude 
    grouping: str
        How the new group should be called
    nside_mthr: int
        nside to use for the calculation of mthr
    mthr_percentile: float
        Percentage used to define the apperent magnitude threhosld
    '''
    
    filled_pixels = np.genfromtxt(os.path.join(outfolder,'filled_pixels.txt')).astype(int)
    with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pixel)),'r+') as cat:
        subcat = cat[grouping]
        subcat.attrs['apparent_magnitude_flag']=apparent_magnitude_flag

        if 'mthr_calculated' not in list(subcat.attrs.keys()):
            subcat.attrs['mthr_calculated'] = False

        if not  subcat.attrs['mthr_calculated']:
        
            ra_central, dec_central = indices2radec(np.array([pixel]),cat.attrs['nside'],nest=cat.attrs['nest'])
            # The big pixel to which it belongs this pixel
            newbig = radec2indeces(ra_central,dec_central,nside_mthr,nest=cat.attrs['nest'])[0]
          
            rafilled, decfilled = indices2radec(filled_pixels,cat.attrs['nside'],nest=cat.attrs['nest'])
            # The big pixels to which all the pixels belong
            skypixmthr = radec2indeces(rafilled,decfilled,nside_mthr,nest=cat.attrs['nest'])
    
            to_read = filled_pixels[skypixmthr==newbig]
    
            m_selected = []
            m_selected = np.append(m_selected,cat['catalog'][apparent_magnitude_flag][cat[grouping]['not_NaN_indices'][:]])
            
            for pix in to_read:
                # If it is the same pixel, just continue
                if pix == pixel:
                    continue
                # It removes the file
                shutil.copyfile(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pix)),
                               os.path.join(outfolder,'pixel_{:d}_{:d}.hdf5'.format(pix,pixel)))
                with h5py.File(os.path.join(outfolder,'pixel_{:d}_{:d}.hdf5'.format(pix,pixel)),'r+') as othercat:
                    m_selected = np.append(
                        m_selected,othercat['catalog'][apparent_magnitude_flag][othercat[grouping]['not_NaN_indices'][:]])
                os.remove(os.path.join(outfolder,'pixel_{:d}_{:d}.hdf5'.format(pix,pixel)))
                
            # If there is no valid galaxy with which to compute the redshift, skip
            if len(m_selected)!=0:
                mthr = np.percentile(m_selected,mthr_percentile)
            else:
                print('Pixel {:d} is empty'.format(pixel))
                mthr = -np.inf
                
            subcat.attrs['mthr_percentile'] = mthr_percentile
            subcat.attrs['nside_mthr'] = nside_mthr
            subcat.attrs['mthr'] = mthr
            # Note that this already include the nans, i.e. np.nan<-np.inf is false
            subcat.create_dataset('brigther_than_mthr',data=cat['catalog'][apparent_magnitude_flag][:]<=mthr)
            subcat.attrs['mthr_calculated'] = True


def get_redshift_grid_for_files(outfolder,pixel,grouping,cosmo_ref,
                               Nintegration=10,Numsigma=3,zcut=None):
    '''
    This function calculates an optimized redshift grid to calculate
    the interpolant for each pixelated file

    Parameters
    ----------
    outfolder: str
        Where are the pixelated files
    pixel: int
        The pixel label you want to read
    grouping: str
        How the new group should be called
    cosmo_ref: class
        Cosmology class for the construnction, should be initialized
    Nintegration: int
        Number of integration points for each likelihood, if array supersede the method
    Numsigma: int
        Number of sigmas for each gaussian likelihood
    zcut: float
        At what redshift to cut the interpolation, if None use cosmo_ref.zmax
    '''
    # If zcut is none, it uses the maximum of the cosmology
    if zcut is None:
        zcut = cosmo_ref.zmax

    if isinstance(Nintegration, np.ndarray):
        if zcut != np.max(Nintegration):
            raise ValueError('The maximum of the grid should be equal to the zcut')
        with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pixel)),'r+') as cat:
            subcat = cat[grouping]
            subcat.attrs['Nintegration']='fixed-array'
            subcat.attrs['Numsigma']=Numsigma
            subcat.attrs['zcut']=zcut
            if 'z_grid_calculated' not in list(subcat.attrs.keys()):
                subcat.attrs['z_grid_calculated'] = False
            if not  subcat.attrs['z_grid_calculated']:
                zmin = cat['catalog']['z'][:]-Numsigma*cat['catalog']['sigmaz'][:]
                zmax = cat['catalog']['z'][:]+Numsigma*cat['catalog']['sigmaz'][:]
                zmin[zmin<np.min(Nintegration)] = np.min(Nintegration)
                zmax[zmax>zcut] = zcut
                # Select all the galaxies for which zmax>zmin and zmax< maximum cosmology redshift
                # zmax<zmin when the galaxy is much beyond zcut.
                valid_galaxies = (zmax>zmin) & (zmax<cosmo_ref.zmax) & subcat['brigther_than_mthr'][:]
                # We put this here as we might be in a situation where a job crashed
                if 'valid_galaxies_interpolant' not in list(subcat.keys()):
                    subcat.create_dataset('valid_galaxies_interpolant',data=valid_galaxies)
                else:
                    del subcat['valid_galaxies_interpolant']
                    subcat.create_dataset('valid_galaxies_interpolant',data=valid_galaxies)
                z_grid = Nintegration
                subcat.create_dataset('z_grid',data=z_grid)
                subcat.attrs['z_grid_calculated'] = True
    else:
        with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pixel)),'r+') as cat:
            subcat = cat[grouping]
            subcat.attrs['Nintegration']=Nintegration
            subcat.attrs['Numsigma']=Numsigma
            subcat.attrs['zcut']=zcut
            if 'z_grid_calculated' not in list(subcat.attrs.keys()):
                subcat.attrs['z_grid_calculated'] = False
            if not  subcat.attrs['z_grid_calculated']:
                # Initialize an array with grid points equally distribuited. Now we are
                # going to populate it
                z_grid=np.linspace(1e-6,zcut,Nintegration)
                #Array with the resolution required between point i and i+1
                resolution_grid=np.ones_like(z_grid)*(zcut-1e-6)/Nintegration  
                zmin = cat['catalog']['z'][:]-Numsigma*cat['catalog']['sigmaz'][:]
                zmax = cat['catalog']['z'][:]+Numsigma*cat['catalog']['sigmaz'][:]
                zmin[zmin<1e-6] = 1e-6
                zmax[zmax>zcut] = zcut
                resolutions = (zmax-zmin)/Nintegration 
                # Select all the galaxies for which zmax>zmin and zmax< maximum cosmology redshift
                # zmax<zmin when the galaxy is much beyond zcut.
                valid_galaxies = (zmax>zmin) & (zmax<cosmo_ref.zmax) & subcat['brigther_than_mthr'][:]
                # We put this here as we might be in a situation where a job crashed
                if 'valid_galaxies_interpolant' not in list(subcat.keys()):
                    subcat.create_dataset('valid_galaxies_interpolant',data=valid_galaxies)
                else:
                    del subcat['valid_galaxies_interpolant']
                    subcat.create_dataset('valid_galaxies_interpolant',data=valid_galaxies)
                idx_sorted = np.argsort(resolutions)
                # This is the index it would sort in decreasing order the resolution needed
                # they corresponds to galaxies
                # Note that if there is no valid galxy, the arrays will stay they are initialized
                idx_sorted = idx_sorted[::-1] 
                for i in idx_sorted:
                    # if a galaxy index is not in the valid galaxies index, skip
                    if not valid_galaxies[i]:
                        continue
                    # These are the points from the old grid falling inside the new grid
                    # These points are not necessary since the new points have a finer resolution
                    to_eliminate = np.where((z_grid>zmin[i]) & (z_grid<zmax[i]))[0]
                    z_grid = np.delete(z_grid,to_eliminate)
                    resolution_grid = np.delete(resolution_grid,to_eliminate)
                    z_integrator = np.linspace(zmin[i],zmax[i],Nintegration)
                    z_grid = np.hstack([z_grid,z_integrator])
                    resolution_grid = np.hstack([resolution_grid,
                                                 np.ones_like(z_integrator)*resolutions[i]/Nintegration])
                    sortme = np.argsort(z_grid)
                    z_grid=z_grid[sortme]
                    resolution_grid=resolution_grid[sortme]
                    z_grid,uind = np.unique(z_grid,return_index=True)
                    resolution_grid = resolution_grid[uind]
                subcat.create_dataset('resolution_grid',data=resolution_grid)

                subcat.create_dataset('z_grid',data=z_grid)
                subcat.attrs['z_grid_calculated'] = True


def initialize_icarogw_catalog(outfolder,outfile,grouping):
    '''
    Iintialize the grouping of the icarogw catalog

    Parameters
    ----------
    outfolder: str
        Where are the pixelated files
    outfile: str
        Name of the icarogw file
    grouping: str
        How the new group should be called
    '''

    filled_pixels = np.genfromtxt(os.path.join(outfolder,'filled_pixels.txt')).astype(int)

    with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(filled_pixels[0])),'r') as tmpcat:
        Nintegration = tmpcat[grouping].attrs['Nintegration']
        zcut = tmpcat[grouping].attrs['zcut']
        with h5py.File(outfile,'a') as icat:

            icat.attrs['nside']=tmpcat.attrs['nside']
            icat.attrs['nest']=tmpcat.attrs['nest']
            icat.attrs['dOmega_sterad']=tmpcat.attrs['dOmega_sterad']
            icat.attrs['dOmega_deg2']=tmpcat.attrs['dOmega_deg2']
            icat.require_group(grouping)

    if Nintegration=='fixed-array':       
        with h5py.File(outfile,'a') as icat:
            actual_filled_pixels = []
            mth_map = []
            for pix in tqdm(filled_pixels,desc='Finding a common redshift grid among pixels'):
                with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pix)),'r') as subcat:
                    z_grid = subcat[grouping]['z_grid'][:] # In this case, z_grid is all the same
                    if np.isfinite(subcat[grouping].attrs['mthr']):
                        actual_filled_pixels.append(pix)
                        mth_map.append(subcat[grouping].attrs['mthr']) 
    else:
        # Initialize an array with grid points equally distribuited. Now we are
        # going to populate it
        z_grid=np.linspace(1e-6,zcut,Nintegration)
        #Array with the resolution required between point i and i+1
        resolution_grid=np.ones_like(z_grid)*(zcut-1e-6)/Nintegration
       
        with h5py.File(outfile,'a') as icat:
            actual_filled_pixels = []
            mth_map = []
            for pix in tqdm(filled_pixels,desc='Finding a common redshift grid among pixels'):
                
                with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pix)),'r') as subcat:
                    z_grid = np.hstack([z_grid,subcat[grouping]['z_grid'][:]])
                    resolution_grid = np.hstack([resolution_grid,subcat[grouping]['resolution_grid'][:]])  
                    if np.isfinite(subcat[grouping].attrs['mthr']):
                        actual_filled_pixels.append(pix)
                        mth_map.append(subcat[grouping].attrs['mthr'])
                
                sortme = np.argsort(z_grid)
                z_grid=z_grid[sortme]
                resolution_grid=resolution_grid[sortme]
                z_grid,uind = np.unique(z_grid,return_index=True)
                resolution_grid = resolution_grid[uind]
    
                to_eliminate = []
                for i in np.arange(1,len(z_grid)-1,1).astype(int):
                    if (resolution_grid[i]>=resolution_grid[i+1]) & (resolution_grid[i]>=resolution_grid[i-1]):
                        to_eliminate.append(i)
                
                z_grid = np.delete(z_grid,to_eliminate)
                resolution_grid = np.delete(resolution_grid,to_eliminate)
                print('Z array is long {:d}'.format(len(z_grid)))
            icat[grouping].create_dataset('resolution_grid',data=resolution_grid)
        
    
    actual_filled_pixels = np.array(actual_filled_pixels)
    mth_map = np.array(mth_map)
    
    with h5py.File(outfile,'a') as icat:
        icat[grouping].create_dataset('z_grid',data=z_grid)    
        
        moc_mthr_map = HealpixMap.moc_from_pixels(icat.attrs['nside'], actual_filled_pixels, 
                               nest = icat.attrs['nest'],density=False)
        
        # The map is initialized with 0 in empty spaces, here we replace with -inf to say complete brightness
        moc_mthr_map[moc_mthr_map.data==0.]=-np.inf
        # This array tells you to what pixel of the filled map it corresponds the filled pixels
        mapping_filled_pixels = np.ones_like(actual_filled_pixels)  
        
        for i,pixo in enumerate(actual_filled_pixels):
            theta,phi = hp.pix2ang(icat.attrs['nside'],pixo,nest=icat.attrs['nest'])
            pix = moc_mthr_map.ang2pix(theta,phi)
            moc_mthr_map[pix] = mth_map[i]
            mapping_filled_pixels[i]=pix
    
            
        # Saves the actually filled helpy pixels
        icat[grouping].create_dataset('mthr_filled_pixels_healpy',data=actual_filled_pixels)
    
        # Saves an array that indicates the filled healpy pixels to which uniq they correspond
        icat[grouping].create_dataset('mthr_filled_pixels_healpy_to_moc_labels',data=mapping_filled_pixels)
    
        # Saves the mthr map
        icat[grouping].create_dataset('mthr_moc_map',data=moc_mthr_map.data)
    
        # Array with uniq identifier
        icat[grouping].create_dataset('uniq_moc_map',data = moc_mthr_map.uniq)

    np.savetxt(os.path.join(outfolder,'{:s}_common_zgrid.txt'.format(grouping)),z_grid)    
        

def calculate_interpolant_files(outfolder,z_grid,pixel,grouping,subgrouping,
                                band,cosmo_ref,epsilon,ptype='gaussian'):
        '''
        This function calculates an optimized redshift grid to calculate
        the interpolant for each pixelated file
    
        Parameters
        ----------
        outfolder: str
            Where are the pixelated files
        z_grid: np.array
            Numpy array with the redshift grid
        pixel: int
            The pixel label you want to read
        grouping: str
            How the new group should be called
        subgrouping: str
            How to call the new group for the interpolant
        band: str
            Electromagnetic band to use, should be a valid icarogw band
        cosmo_ref: class
            Cosmology class for the construnction, should be initialized
        epsilon: float
            Exponent of the luminosity weight
        ptype: str
            Type of likelihood for the EM side.
        '''
       
        calc_kcorr=kcorr(band)
        sch_fun=galaxy_MF(band=band)
        sch_fun.build_MF(cosmo_ref)
        sch_fun.build_effective_number_density_interpolant(epsilon)

        absM_rate=log_powerlaw_absM_rate(epsilon=epsilon)

        with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pixel)),'r+') as cat:
            subcat=cat[grouping].require_group(subgrouping)
            subcat.attrs['band'] = band

            if 'interpolant_calculated' not in list(subcat.attrs.keys()):
                subcat.attrs['interpolant_calculated'] = False

            if not  subcat.attrs['interpolant_calculated']:            
                subcat.attrs['epsilon']=epsilon
                subcat.attrs['ptype']=ptype   
                interpo = np.zeros_like(z_grid)
                Ngalaxies = len(cat['catalog']['z'][:])
                toloop = np.where(cat[grouping]['valid_galaxies_interpolant'][:])[0]
                for j in toloop:
                    Mv=m2M(cat['catalog'][cat[grouping].attrs['apparent_magnitude_flag']][j],cosmo_ref.z2dl(z_grid),calc_kcorr(z_grid))  
                    interpo+=absM_rate.evaluate(sch_fun,Mv)*EM_likelihood_prior_differential_volume(z_grid,
                                                                cat['catalog']['z'][j],
                                                                cat['catalog']['sigmaz'][j],cosmo_ref,Numsigma=cat[grouping].attrs['Numsigma'],ptype=ptype)/cat.attrs['dOmega_sterad'] 
                    # We can divide by the old grid dOmega_sterad as full pixels are all of same size
        
                subcat.create_dataset('vals_interpolant',data=interpo)
                subcat.attrs['interpolant_calculated'] = True

class  icarogw_catalog(object):
    '''
    This is the class used to handle the icarogw catalog

    Parameters
    ----------
    outfile: str
        The original icarogw file
    grouping: str
        What group to use for the analysis, i.e. the EM band used
    subgrouping: str
        What case to use for the galaxy weights
    '''
    
    def __init__(self,outfile,grouping,subgrouping):
        self.outfile = outfile
        self.grouping = grouping
        self.subgrouping = subgrouping
        
    def build_from_pixelated_files(self,outfolder):
        '''
        Build the catalog file from the pixelated files

        Parameters
        ----------
        outfolder: str
            Path where the pixelated files can be found
        '''

        with h5py.File(self.outfile,'r') as icat:
            self.moc_mthr_map = HealpixMap(data=icat[self.grouping]['mthr_moc_map'][:],uniq=icat[self.grouping]['uniq_moc_map'][:])
            self.z_grid = icat[self.grouping]['z_grid'][:]
            # Saves the actually filled helpy pixels
            filled_pixels = icat[self.grouping]['mthr_filled_pixels_healpy'][:]
            # Saves an array that indicates the filled healpy pixels to which uniq they correspond
            moc_pixels = icat[self.grouping]['mthr_filled_pixels_healpy_to_moc_labels'][:]

        # Sorted uniq grid
        self.sky_grid = np.sort(self.moc_mthr_map.uniq)
        dNgal_dzdOm_vals = []
        for pix in tqdm(self.sky_grid,desc='Bulding sky grid'):
            idx = np.where(moc_pixels == pix)[0]
            if len(idx) == 0:
                dNgal_dzdOm_vals.append(np.zeros_like(self.z_grid))
            else:
                with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(filled_pixels[idx[0]]))) as pcat:
                    dNgal_dzdOm_vals.append(pcat[self.grouping][self.subgrouping]['vals_interpolant'][:])
                    band = pcat[self.grouping][self.subgrouping].attrs['band']
                    epsilon = pcat[self.grouping][self.subgrouping].attrs['epsilon']
        
        self.band = band
        self.epsilon = epsilon
        self.calc_kcorr=kcorr(band)
        self.sch_fun=galaxy_MF(band=band)
        self.sch_fun.build_effective_number_density_interpolant(epsilon)
        self.dNgal_dzdOm_vals = np.column_stack(dNgal_dzdOm_vals)

    def save_to_hdf5_file(self):
        '''
        Saves the interpolants and everything neeeded in a single hdf5 file
        '''

        with h5py.File(self.outfile,'a') as icat:
            subgroup = icat[self.grouping].require_group(self.subgrouping)
            subgroup.attrs['epsilon'] = self.epsilon
            subgroup.attrs['band'] = self.band
            subgroup.create_dataset('vals_interpolant',data=self.dNgal_dzdOm_vals)

    def load_from_hdf5_file(self):
        '''
        Load the catalog and everything needed
        '''
        
        with h5py.File(self.outfile,'r') as icat:
            self.moc_mthr_map = HealpixMap(data=icat[self.grouping]['mthr_moc_map'][:],uniq=icat[self.grouping]['uniq_moc_map'][:])
            self.z_grid = icat[self.grouping]['z_grid'][:]
            self.band = icat[self.grouping][self.subgrouping].attrs['band']
            self.epsilon = icat[self.grouping][self.subgrouping].attrs['epsilon']
            self.dNgal_dzdOm_vals = icat[self.grouping][self.subgrouping]['vals_interpolant'][:]
         
        self.calc_kcorr=kcorr(self.band)
        self.sch_fun=galaxy_MF(band=self.band)
        self.sch_fun.build_effective_number_density_interpolant(self.epsilon)           
        # Sorted uniq grid
        self.sky_grid = np.sort(self.moc_mthr_map.uniq)
        
    def get_NUNIQ_pixel(self,ra,dec):
        '''
        Gets the MOC map pixels from RA and dec

        Parameters
        ----------
        ra, dec: np.array
            Right Ascension and declination in radians

        Returns
        -------
        uniq pixel of the MOC map
        '''
        return self.moc_mthr_map.ang2pix(np.pi/2-dec,ra) 

    def calc_Mthr(self,z,radec_indices,cosmology,dl=None):
        ''' 
        This function returns the Absolute magnitude threshold calculated from the apparent magnitude threshold
        
        Parameters
        ----------
        z: xp.array
            Redshift
        radec_indices: xp.array
            Healpy indices
        cosmology: class 
            cosmology class
        dl: xp.array
            dl values already calculated
        
        Returns
        -------
        Mthr: xp.array
            Apparent magnitude threshold
        '''
        
        # RADEC insidec must be in moc map

        if dl is None:
            dl=cosmology.z2dl(z)
        
        mthr_arrays=self.moc_mthr_map[radec_indices]
        return m2M(mthr_arrays,dl,self.calc_kcorr(z))

    def effective_galaxy_number_interpolant(self,z,skypos,cosmology,dl=None):
        '''
        Returns an evaluation of dNgal/dzdOmega, it requires `calc_dN_by_dzdOmega_interpolant` to be called first.
        It needs the schecter function to be updated
        
        Parameters
        ----------
        z: xp.array
            Redshift array
        skypos: xp.array
            Array containing the healpix indeces where to evaluate the interpolant
        cosmology: class
            cosmology class to use for the computations
        dl: xp.array
            Luminosity distance in Mpc
        average: bool
            Use the sky averaged differential of effective number of galaxies in each pixel
        '''
        
        xp=get_module_array(z)
        sx=get_module_array_scipy(z)
       
        originshape=z.shape
        z=z.flatten()
        skypos=skypos.flatten()
        
        if dl is None:
            dl=cosmology.z2dl(z)
        dl=dl.flatten()
        
        z_grid = self.z_grid
        dNgal_dzdOm_vals = self.dNgal_dzdOm_vals
        pixel_grid = self.sky_grid
    
        Mthr_array=self.calc_Mthr(z,skypos,cosmology,dl=dl)
        # Baiscally tells that if you are above the maximum interpolation range, you detect nothing
        Mthr_array[z>z_grid[-1]]=-xp.inf
        
        gcpart=sx.interpolate.interpn((z_grid,pixel_grid),dNgal_dzdOm_vals,xp.column_stack([z,skypos]),bounds_error=False,
                            fill_value=0.,method='linear') # If a posterior samples fall outside, then you return 0
        
        bgpart=self.sch_fun.background_effective_galaxy_density(Mthr_array,z)*cosmology.dVc_by_dzdOmega_at_z(z)
        
        return gcpart.reshape(originshape),bgpart.reshape(originshape)
        
    def check_differential_effective_galaxies(self,z,radec_indices_list,cosmology):
        '''
        This method checks the comoving volume distribution built from the catalog. It is basically a complementary check to the galaxy schecther function
        distribution
        
        Parameters
        ----------
        z: np.array
            Array of redshifts where to evaluate the effective galaxy density
        radec_indices: np.array
            Array of pixels on which you wish to average the galaxy density
        cosmology: class
            cosmology class to use
            
        Returns
        -------
        gcp: np.array
            Effective galaxy density from the catalog
        bgp: np.array
            Effective galaxy density from the background correction
        inco: np.array
            Incompliteness array
        fig: object
            Handle to the figure object
        ax: object
            Handle to the axis object
        '''
        gcp,bgp,inco=np.zeros([len(z),len(radec_indices_list)]),np.zeros([len(z),len(radec_indices_list)]),np.zeros([len(z),len(radec_indices_list)])
        
        for i,skypos in enumerate(radec_indices_list):
            gcp[:,i],bgp[:,i]=self.effective_galaxy_number_interpolant(z,skypos*np.ones_like(z).astype(int),cosmology)
            Mthr_array=self.calc_Mthr(z,np.ones_like(z,dtype=int)*skypos,cosmology)
            Mthr_array[z>self.z_grid[-1]]=-np.inf
            inco[:,i]=self.sch_fun.background_effective_galaxy_density(Mthr_array,z)/self.sch_fun.background_effective_galaxy_density(-np.ones_like(Mthr_array)*np.inf,z)
            
        fig,ax=plt.subplots(2,1,sharex=True)
        
        theo=self.sch_fun.background_effective_galaxy_density(-np.inf*np.ones_like(z),z)*cosmology.dVc_by_dzdOmega_at_z(z)
                
        ax[0].fill_between(z,np.percentile(gcp,5,axis=1),np.percentile(gcp,95,axis=1),color='limegreen',alpha=0.2)
        ax[0].plot(z,np.median(gcp,axis=1),label='Catalog part',color='limegreen',lw=2)
        
        ax[0].plot(z,np.median(bgp,axis=1),label='Background part',color='slateblue',lw=2)
        
        ax[0].fill_between(z,np.percentile(bgp+gcp,5,axis=1),np.percentile(bgp+gcp,95,axis=1),color='tomato',alpha=0.2)
        ax[0].plot(z,np.median(bgp+gcp,axis=1),label='Sum',color='tomato',lw=2)        
        
        ax[0].plot(z,theo,label='Theoretical',color='k',lw=2,ls='--')
        ax[0].set_ylim([10,1e7])
        
        ax[0].legend()
        
        ax[0].set_yscale('log')
                
        ax[1].fill_between(z,np.percentile(1-inco,5,axis=1),np.percentile(1-inco,95,axis=1),color='dodgerblue',alpha=0.5)
        ax[1].plot(z,np.median(1-inco,axis=1),label='Completeness',color='dodgerblue',lw=1)
        ax[1].legend()
        
        return gcp,bgp,inco,fig,ax
        
# ---------------------------------------------------------------------------------------

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
        if self.band not in ['W1-glade+','K-glade+','bJ-glade+','W1-upglade','g-upglade']:
            raise ValueError('Band not known please use either {:s}'.format(' '.join(['W1-glade+','K-glade+','bJ-glade+',
                                                                                     'W1-upglade','g-upglade'])))
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
        if self.band == 'W1-glade+':
            k_corr = -1*(4.44e-2+2.67*z+1.33*(z**2.)-1.59*(z**3.)) #From Maciej email
        elif self.band == 'K-glade+':
            # https://iopscience.iop.org/article/10.1086/322488/pdf 4th page lhs
            to_ret=-6.0*xp.log10(1+z)
            to_ret[z>0.3]=-6.0*xp.log10(1+0.3)
            k_corr=-6.0*xp.log10(1+z)
        elif self.band == 'bJ-glade+':
            # Fig 5 caption from https://arxiv.org/pdf/astro-ph/0111011.pdf
            # Note that these corrections also includes evolution corrections
            k_corr=(z+6*xp.power(z,2.))/(1+15.*xp.power(z,3.))
        elif (self.band == 'W1-upglade') | (self.band == 'g-upglade'):
            # In upglade k-corrections are already applied
            k_corr = xp.zeros_like(z) 
        return k_corr

# LVK Reviewed
def user_normal(x,mu,sigma):
    ''' 
    A utility function meant only for this module. It returns a normalized gaussian distribution
    
    Parameters
    ----------
    x, mu, sigma: xp.arrays
        Points at which to evaluate the gaussian with mean mu and std sigma
    
    Returns
    -------
    Values
    
    '''
    xp=get_module_array(x)
    return xp.power(2*xp.pi*(sigma**2),-0.5)*xp.exp(-0.5*xp.power((x-mu)/sigma,2.))

# LVK Reviewed
def EM_likelihood_prior_differential_volume(z,zobs,sigmaz,cosmology,Numsigma=1.,ptype='uniform'):
    ''' 
    A utility function meant only for this module. Calculates the EM likelihood in redshift times a uniform in comoving volume prior
    
    Parameters
    ----------
    z: xp.array
        Values at which to evaluate the EM likelihood times the prior. This is usually and array that starts from 0 and goes to zcut
    zobs: float 
        Central value of the galaxy redshift
    sigmaobs: float
        Std of galaxy redshift localization. Note if flat EM likelihood sigma is the half-widht of the box distribution.
    cosmology: Class
        Cosmology class from icarogw
    Numsigma: float
        Half Width for the uniform distribution method in terms of sigmaz
    ptype: string
        Type of EM likelihood, ''uniform'' for uniform distribution, ''gaussian'' for gaussian
    
    Returns
    -------
    Values of the EM likelihood times the prior evaluated on z
    
    '''
    xp=get_module_array(z)
    # Lower limit for the integration. A galaxy must be at a positive redshift
    zvalmin=xp.array([1e-6,zobs-Numsigma*sigmaz]).max()
    #zvalmax=xp.array([z.max(),zobs+Numsigma*sigmaz]).min()    
    
    if ptype=='uniform':
        
        # higher limit for the integration. If it is localized  partialy above z_cut, it counts less
        zvalmax=np.minimum(zobs+5.*sigmaz,cosmology.zmax)
        if zvalmax<=zvalmin:
            return xp.zeros_like(z)
    
        prior_eval=4*xp.pi*cosmology.dVc_by_dzdOmega_at_z(z)*((z>=(zobs-Numsigma*sigmaz)) & (z<=(zobs+Numsigma*sigmaz)))/(cosmology.z2Vc(zvalmax)-cosmology.z2Vc(zvalmin))
    elif ptype=='gaussian':
        
        zvalmax=np.minimum(zobs+5.*sigmaz,cosmology.zmax)
        if zvalmax<=zvalmin:
            return xp.zeros_like(z)
    
        prior_eval=cosmology.dVc_by_dzdOmega_at_z(z)*user_normal(z,zobs,sigmaz)
        zproxy=xp.linspace(zvalmin,zvalmax,5000)
        normfact=xp.trapz(cosmology.dVc_by_dzdOmega_at_z(zproxy)*user_normal(zproxy,zobs,sigmaz),zproxy)
        
        if normfact==0.:
            print(zobs,sigmaz)
            raise ValueError('Normalization failed')
            
        if np.isnan(normfact):
            print(zobs,sigmaz)
            raise ValueError('Normalization failed')
            
        prior_eval/=normfact
        
    elif ptype=='gaussian_nocom':
        
        zvalmax=np.minimum(zobs+5.*sigmaz,cosmology.zmax)
        if zvalmax<=zvalmin:
            return xp.zeros_like(z)
    
        prior_eval=user_normal(z,zobs,sigmaz)
        zproxy=xp.linspace(zvalmin,zvalmax,5000)
        normfact=xp.trapz(user_normal(zproxy,zobs,sigmaz),zproxy)
        
        if normfact==0.:
            print(zobs,sigmaz)
            raise ValueError('Normalization failed')
            
        if np.isnan(normfact):
            print(zobs,sigmaz)
            raise ValueError('Normalization failed')
            
        prior_eval/=normfact

    return prior_eval


    
    
    
    
    
    
