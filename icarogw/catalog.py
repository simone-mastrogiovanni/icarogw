from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn, is_there_cupy
from .conversions import radec2indeces, indices2radec, M2m, m2M
from .cosmology import galaxy_MF, log_powerlaw_absM_rate, astropycosmology
from astropy.cosmology import Planck15

import healpy as hp
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import os
import shutil
from mhealpy import HealpixBase, HealpixMap
import mpmath

LOWERL=np.nan_to_num(-np.inf)

#LVK reviewed
def create_pixelated_catalogs(outfolder,nside,groups_dict,fields_to_take=None,batch=100000,nest=False):
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
    fields_to_take: list of strings
        They are the list of variables you would like to save or append to the pixelated files.
        If this keyword is None, then it means you are creating the  files for the first time
    batch: int
        How many galaxies to process
    nest: bool
        Nest flag for healpy
    '''
    list_of_keys = list(groups_dict.keys())
    if fields_to_take == None:
        fields_to_take = list_of_keys

    # Loops over the pixels and create fiels if they do not exist, otherwise open them in append mode
    for i in tqdm(range(hp.nside2npix(nside)),desc='Creating the file with pixels fields'):    
        cat = h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(i)),'a')
        # Create catalog group if it does not exist
        catobj = cat.require_group('catalog')
        cat.attrs['nside']=nside
        cat.attrs['nest']=nest
        cat.attrs['dOmega_sterad']=hp.nside2pixarea(nside,degrees=False)
        cat.attrs['dOmega_deg2']=hp.nside2pixarea(nside,degrees=True)
        cat.attrs['Ntotal_galaxies_original']=0
        for key in list_of_keys:
            # Creates the field if it does not exist
            if key not in list(catobj.keys()): 
                catobj.create_dataset(key, data=np.array([]), compression="gzip", chunks=True, maxshape=(None,))
        cat.close()

    # Creates the checkpoint file if it does not exist
    if not os.path.isfile(os.path.join(outfolder,'checkpoint_creation.txt')):
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

                # If the fields to take is equal to the list of keys of the galaxy catalog
                # It means you are writing the file for the first time, so you cound galaxies
                # Note that Ntotal_galaxies_original contains all the galaxies even with ones with NaNs
                if fields_to_take == list_of_keys:
                    pp.attrs['Ntotal_galaxies_original']+=len(galaxies_id)
                    
                # The loop is only on the fields to take as you might want to add fields when the file is created
                for key in fields_to_take:
                    pix[key].resize((pix[key].shape[0] + len(galaxies_id)), axis = 0)
                    pix[key][-len(galaxies_id):] = groups_dict[key][istart:istart+batch][galaxies_id]
        istart+=batch
        pbar.update(batch)
        np.savetxt(os.path.join(outfolder,'checkpoint_creation.txt'),np.array([istart]),fmt='%d')
    pbar.close()

    # Reset the checkpoint creation as we might want to add more fields later
    istart = 0
    np.savetxt(os.path.join(outfolder,'checkpoint_creation.txt'),np.array([istart]),fmt='%d')    

#LVK reviewed
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

#LVK reviewed
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
        
#LVK reviewed
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


#LVK reviewed
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
        # zcut might no deviate from the maximum of the redshift grid by 1e-4
        if np.abs(zcut-np.max(Nintegration))>=1e-4:
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
                # zmax<zmin when the galaxy is much beyond zcut or the galaxy is at redshift<-1
                # The condition (zmax<cosmo_ref.zmax) is always satisfied if zcut<cosmo_ref.zmax
                # The and condition is also with not_NaN_indices as galaxies should have all the entries to be used for interpolant
                valid_galaxies = (zmax>zmin) & (zmax<cosmo_ref.zmax) & subcat['brigther_than_mthr'][:] & subcat['not_NaN_indices'][:]
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
                # The condition (zmax<cosmo_ref.zmax) is always satisfied if zcut<cosmo_ref.zmax
                # The and condition is also with not_NaN_indices as galaxies should have all the entries to be used for interpolant
                valid_galaxies = (zmax>zmin) & (zmax<cosmo_ref.zmax) & subcat['brigther_than_mthr'][:] & subcat['not_NaN_indices'][:]
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


#LVK reviewed
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
            subicat = icat.require_group(grouping)
            subicat.attrs['zcut'] = tmpcat[grouping].attrs['zcut']
            subicat.attrs['zcut'] = tmpcat[grouping].attrs['Nintegration']
            

    if Nintegration=='fixed-array':       
        with h5py.File(outfile,'a') as icat:
            actual_filled_pixels = []
            mth_map = []
            for pix in tqdm(filled_pixels,desc='Finding a common redshift grid among pixels'):
                with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(pix)),'r') as subcat:
                    
                    # If there is at least a galaxy in the pixel valid for the interpolant
                    # this variable is true
                    valid_gal = np.any(subcat[grouping]['valid_galaxies_interpolant'][:])

                    # If no galaxy is valid for the interpolant, goes to the next pixel
                    if valid_gal:
                        pass
                    else:
                        continue
                    
                    z_grid = subcat[grouping]['z_grid'][:] # In this case, z_grid is all the same
                    # Extra check to see if the pixel is empty
                    if np.isfinite(subcat[grouping].attrs['mthr']):
                        actual_filled_pixels.append(pix)
                        mth_map.append(subcat[grouping].attrs['mthr']) 
    
    # TO-DO: Optimize this method as for deep galaxy catalogs it is too demanding
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
    
        # Saves an array that indicates the filled healpy pixels to which pixel label they correspond on the non uniform map
        icat[grouping].create_dataset('mthr_filled_pixels_healpy_to_moc_labels',data=mapping_filled_pixels)
    
        # Saves the mthr map
        icat[grouping].create_dataset('mthr_moc_map',data=moc_mthr_map.data)
    
        # Array with uniq identifier
        icat[grouping].create_dataset('uniq_moc_map',data = moc_mthr_map.uniq)

    np.savetxt(os.path.join(outfolder,'{:s}_common_zgrid.txt'.format(grouping)),z_grid)    
        

#LVK reviewed
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
                    # A patch to use K-corrections from upGLADE
                    if band=='g-upglade':
                        kcorr_arr = calc_kcorr(z_grid, k0 = cat['catalog']['K_g'][j] , dkbydz = cat['catalog']['dKbydz_g'][j] ,z0 = cat['catalog']['z'][j])
                    elif band=='r-upglade':
                        kcorr_arr = calc_kcorr(z_grid, k0 = cat['catalog']['K_r'][j] , dkbydz = cat['catalog']['dKbydz_r'][j] ,z0 = cat['catalog']['z'][j])
                    elif band=='W1-upglade':
                        kcorr_arr = calc_kcorr(z_grid, k0 = cat['catalog']['K_W1'][j] , dkbydz = cat['catalog']['dKbydz_W1'][j] ,z0 = cat['catalog']['z'][j])
                    else:
                        kcorr_arr = calc_kcorr(z_grid)
                        
                    Mv=m2M(cat['catalog'][cat[grouping].attrs['apparent_magnitude_flag']][j],cosmo_ref.z2dl(z_grid),kcorr_arr)                      
                    interpo+=absM_rate.evaluate(sch_fun,Mv)*EM_likelihood_prior_differential_volume(z_grid,
                                                                cat['catalog']['z'][j],
                                                                cat['catalog']['sigmaz'][j],cosmo_ref,Numsigma=cat[grouping].attrs['Numsigma'],ptype=ptype)/cat.attrs['dOmega_sterad'] 

                    # An additional check to see if something is going wrong.
                    if np.isnan(interpo).all():
                        print(j)
                        print('dk',cat['catalog']['dKbydz_g'][j])
                        print('k',cat['catalog']['K_g'][j])
                        print('Mv', Mv)
                        print('m', cat['catalog'][cat[grouping].attrs['apparent_magnitude_flag']][j])
                        print('z', cat['catalog']['z'][j])
                        print('sigmaz', cat['catalog']['sigmaz'][j])

                        raise ValueError('All NANS')

                
                    # We can divide by the old grid dOmega_sterad as full pixels are all of same size
        
                subcat.create_dataset('vals_interpolant',data=interpo)
                subcat.attrs['interpolant_calculated'] = True


#LVK reviewed
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
            # Saves an array that indicates the filled healpy pixels to which pixel label they correspond
            moc_pixels = icat[self.grouping]['mthr_filled_pixels_healpy_to_moc_labels'][:]
            
        # That indentifies the index of the pixels on the mthr map in sorted order
        self.sky_grid = np.arange(0,len(self.moc_mthr_map.data),1).astype(int)
        dNgal_dzdOm_vals = []
        bg_vals = []
        loaded_sch = False
        for pix in tqdm(self.sky_grid,desc='Bulding sky grid'):
            idx = np.where(moc_pixels == pix)[0]
            if len(idx) == 0:
                dNgal_dzdOm_vals.append(np.zeros_like(self.z_grid))
                bg_vals.append(self.sch_fun.background_effective_galaxy_density(-np.inf*np.ones_like(self.z_grid),self.z_grid)
                    *cosmology_proxy.dVc_by_dzdOmega_at_z(self.z_grid))
            else:
                with h5py.File(os.path.join(outfolder,'pixel_{:d}.hdf5'.format(filled_pixels[idx[0]]))) as pcat:
                    dNgal_dzdOm_vals.append(pcat[self.grouping][self.subgrouping]['vals_interpolant'][:])
                    if not loaded_sch:                
                        band = pcat[self.grouping][self.subgrouping].attrs['band']
                        epsilon = pcat[self.grouping][self.subgrouping].attrs['epsilon']
                        self.band = band
                        self.epsilon = epsilon
                        self.calc_kcorr=kcorr(band)
                        self.sch_fun=galaxy_MF(band=band)
                        self.sch_fun.build_effective_number_density_interpolant(epsilon)
                        # Initialize a cosmology with zmax at double the distance
                        cosmology_proxy = astropycosmology(zmax=self.z_grid[-1]*2)
                        cosmology_proxy.build_cosmology(Planck15)
                        self.sch_fun.build_MF(cosmology_proxy)
                        dl_proxy=cosmology_proxy.z2dl(self.z_grid)
                        loaded_sch = True

                    # Mthr array as function of redshift in pixel
                    Mthr_array=self.calc_Mthr(self.z_grid,pix*np.ones_like(self.z_grid).astype(int),cosmology_proxy,dl=dl_proxy)
                    bg_vals.append(self.sch_fun.background_effective_galaxy_density(Mthr_array,self.z_grid)*cosmology_proxy.dVc_by_dzdOmega_at_z(self.z_grid))
        
        self.dNgal_dzdOm_vals = np.column_stack(dNgal_dzdOm_vals)
        self.bg_vals = np.column_stack(bg_vals)

    def make_me_empty(self):
        self.moc_mthr_map._data = -np.inf*np.ones_like(self.moc_mthr_map._data)
        self.dNgal_dzdOm_vals = np.zeros_like(self.dNgal_dzdOm_vals)
        self.dNgal_dzdOm_vals_av = np.zeros_like(self.dNgal_dzdOm_vals_av)
        cosmology_proxy = astropycosmology(zmax=self.z_grid[-1]*2)
        cosmology_proxy.build_cosmology(Planck15)
        self.sch_fun.build_MF(cosmology_proxy)
        self.bg_vals_av = self.sch_fun.background_effective_galaxy_density(-np.inf*np.ones_like(self.z_grid),self.z_grid)*cosmology_proxy.dVc_by_dzdOmega_at_z(self.z_grid)
        

    def save_to_hdf5_file(self):
        '''
        Saves the interpolants and everything neeeded in a single hdf5 file
        '''

        with h5py.File(self.outfile,'a') as icat:
            subgroup = icat[self.grouping].require_group(self.subgrouping)
            subgroup.attrs['epsilon'] = self.epsilon
            subgroup.attrs['band'] = self.band
            subgroup.create_dataset('vals_interpolant',data=self.dNgal_dzdOm_vals)
            subgroup.create_dataset('bg_vals_interpolant',data=self.bg_vals)

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
            self.bg_vals = icat[self.grouping][self.subgrouping]['bg_vals_interpolant'][:]
         
        self.calc_kcorr=kcorr(self.band)
        self.sch_fun=galaxy_MF(band=self.band)
        self.sch_fun.build_effective_number_density_interpolant(self.epsilon)           
        # Sorted uniq grid
        self.sky_grid = np.arange(0,len(self.moc_mthr_map.data),1).astype(int)

        # Sky averaged in-catalog part
        self.dNgal_dzdOm_vals_av = np.mean(self.dNgal_dzdOm_vals,axis=1)
        self.bg_vals_av = np.mean(self.bg_vals,axis=1)
        # Deleting as this is not necessary
        del self.bg_vals

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
        xp = get_module_array(mthr_arrays)
        # Once the K-corrections are taken into account in the construction of the 
        # LOS prior we do not need to account them anymore
        return m2M(mthr_arrays,dl, xp.zeros_like(dl))

    def effective_galaxy_number_interpolant(self,z,skypos,cosmology,average=False):
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
            cosmology class to use for the computations, cosmology must be background cosmology, not MOD GRAVITY
        average: bool
            Use the sky averaged differential of effective number of galaxies in each pixel
        '''
        
        xp=get_module_array(z)
        sx=get_module_array_scipy(z)
       
        originshape=z.shape
        z=z.flatten()
        skypos=skypos.flatten()
        # This is the EM dl, it corresponds to GW if not mod gravity
        dl=cosmology.z2dl(z)
        dl=dl.flatten()
        
        z_grid = self.z_grid
        dNgal_dzdOm_vals = self.dNgal_dzdOm_vals
        pixel_grid = self.sky_grid
    
        Mthr_array=self.calc_Mthr(z,skypos,cosmology,dl=dl)
        # Baiscally tells that if you are above the maximum interpolation range or below, you detect nothing
        # By definition
        idx_out = (z>z_grid[-1]) | (z<z_grid[0])
        Mthr_array[idx_out]=-xp.inf
        
        if average:
            # The zero is there to indicate that if you fall outside
            # the interpolant is 0, as in the sky-dependent part
            gcpart=xp.interp(z,z_grid,self.dNgal_dzdOm_vals_av,left=0.,right=0.)

            # If you are outside the redshift interpolantion range, the values are replaced later
            bgpart=xp.interp(z,z_grid,self.bg_vals_av,left=self.bg_vals_av[0],right=self.bg_vals_av[-1])
            # the values outside the interpolantion range have a backround given by the out-of-catalog
            if xp.any(idx_out):
                # It puts totally an out of catalog
                bgpart[idx_out] = self.sch_fun.background_effective_galaxy_density(Mthr_array[idx_out],z[idx_out])*cosmology.dVc_by_dzdOmega_at_z(z[idx_out])
        else:
            gcpart=sx.interpolate.interpn((z_grid,pixel_grid),dNgal_dzdOm_vals,xp.column_stack([z,skypos]),bounds_error=False,
                                fill_value=0.,method='linear') # If a posterior samples fall outside, then you return 0

            # The values outside interpolation range have an out-of-catalog given by the full completeness correction
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


#############################################
# A set of function from gwcosmo below to query the gwcosmo 
# galaxy catalogs

def gwcosmo_get_offset(LOS_catalog):
    diction = eval(LOS_catalog.attrs['opts'])
    return diction["offset"]

def gwcosmo_get_z_array(LOS_catalog):
    return LOS_catalog['z_array'][:]

def gwcosmo_get_array(LOS_catalog, arr_name):
    offset = gwcosmo_get_offset(LOS_catalog)

    arr = LOS_catalog[str(arr_name)][:]
    arr = np.exp(arr)
    arr -= offset

    return arr

def gwcosmo_get_empty_catalog(LOS_catalog):
    return gwcosmo_get_array(LOS_catalog, "empty_catalogue")

def gwcosmo_get_zprior_full_sky(LOS_catalog):
    return gwcosmo_get_array(LOS_catalog, "combined_pixels")

def gwcosmo_get_zprior(LOS_catalog, pixel_index):
    return gwcosmo_get_array(LOS_catalog, str(pixel_index))


class gwcosmo_catalog(object):
    def __init__(self,gwcosmo_file,nside,band,epsilon):
        self.band = band
        self.epsilon = epsilon
        self.nside = nside
        self.sch_fun=galaxy_MF(band=self.band)
        self.sch_fun.build_effective_number_density_interpolant(self.epsilon)
        self.sky_grid = np.arange(0,hp.nside2npix(nside),1).astype(int)
        with h5py.File(gwcosmo_file,'r') as gwcosmo:
            self.z_grid = gwcosmo_get_z_array(gwcosmo)
            self.pz_empty = gwcosmo_get_empty_catalog(gwcosmo)
            self.dNgal_dzdOm_vals_av = gwcosmo_get_zprior_full_sky(gwcosmo)
            self.dNgal_dzdOm_vals = []
            for ipix in self.sky_grid:
                self.dNgal_dzdOm_vals.append(gwcosmo_get_zprior(gwcosmo,ipix))
        self.dNgal_dzdOm_vals = np.column_stack(self.dNgal_dzdOm_vals)
        
    def make_me_empty(self):
        self.dNgal_dzdOm_vals_av = self.pz_empty
        self.dNgal_dzdOm_vals = np.column_stack([self.pz_empty for i in range(hp.nside2npix(self.nside))])
        
    def get_NUNIQ_pixel(self,ra,dec):
        return hp.ang2pix(self.nside,np.pi/2-dec,ra,nest=True) 

    def effective_galaxy_number_interpolant(self,z,skypos,cosmology,dl=None,average=False): 
        '''
        Note that everything here is in the in-catalog part
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
        
        if average:
            gcpart=xp.interp(z,z_grid,self.dNgal_dzdOm_vals_av,left=0.,right=0.)
            bgpart=xp.zeros_like(gcpart)           
        else:
            gcpart=sx.interpolate.interpn((z_grid,pixel_grid),dNgal_dzdOm_vals,xp.column_stack([z,skypos]),bounds_error=False,
                                fill_value=0.,method='linear') 
            bgpart=xp.zeros_like(gcpart)           

        return gcpart.reshape(originshape),bgpart.reshape(originshape)



#############################################

# ---------------------------------------------------------------------------------------

#LVK reviewed
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
        if self.band not in ['W1-glade+','K-glade+','bJ-glade+','W1-upglade','g-upglade','r-upglade']:
            raise ValueError('Band not known please use either {:s}'.format(' '.join(['W1-glade+','K-glade+','bJ-glade+',
                                                                                     'W1-upglade','g-upglade','r-upglade'])))
    def __call__(self,z, k0 = None, dkbydz=None, z0 = None):
        '''
        Evaluates the K-corrections at a given redshift, See Eq. 2 of https://arxiv.org/abs/astro-ph/0210394
        
        Parameters
        ----------
        z: xp.array
            Redshift
        k0: xp.array
            Array of K-corrections computed for a z0 (only used with upglade)
        dkbydz: xp.array
            Array of K-corrections derivatives
        z0: xp.array
            Redshift at which the K-correction is computed
            
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
            # Fig 8 caption from https://arxiv.org/pdf/astro-ph/0111011.pdf
            # Note that these corrections also includes evolution corrections
            k_corr=(z+6*xp.power(z,2.))/(1+20.*xp.power(z,3.))
        elif (self.band == 'W1-upglade') | (self.band == 'g-upglade') | (self.band == 'r-upglade'):
            k_corr = k0+dkbydz*(z-z0)
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
def EM_likelihood_prior_differential_volume(z,zobs,sigmaz,cosmology,Numsigma=3.,ptype='uniform'):
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
    
# LVK Reviewed
class galaxy_catalog(object):
    '''
    A class to handle galaxy catalogs. This class creates a hdf5 file containing all the necessary informations.
    '''
    
    
    def __init__(self):
        pass
    
    def create_hdf5(self,filename,cat_data,band,nside):
        '''
        Creates the HDF5 file

        Parameters
        ----------
        filename: string
            HDF5 file name to create
        cat_data: dictionary
            Dictionary of arrays containings for each galaxy 'ra': right ascensions in rad, 'dec': declination in radians
            'z': galaxy redshift, 'sigmaz': redshift uncertainty (can not be zero), 'm': apparent magnitude.
        band: string
            Band to use for the background corrections, need to be compatible with apparent magnitude. Bands available
            'K', 'W1', 'bJ'
        nside: int
            Nside to use for the healpy pixelization
        '''
        
        # This for loop removes the galaxies with NaNs or inf as entries 
        for key in list(cat_data.keys()):
            tokeep=np.where(np.isfinite(cat_data[key]))[0]
            cat_data={subkey:cat_data[subkey][tokeep] for subkey in list(cat_data.keys())}
        
        # Pixelize the galaxies
        cat_data['sky_indices'] = radec2indeces(cat_data['ra'],cat_data['dec'],nside)
        
        with h5py.File(filename,'w-') as f:

            cat=f.create_group('catalog')
            cat.attrs['band']=band
            cat.attrs['nside']=nside
            cat.attrs['npixels']=hp.nside2npix(nside)
            cat.attrs['dOmega_sterad']=hp.nside2pixarea(nside,degrees=False)
            cat.attrs['dOmega_deg2']=hp.nside2pixarea(nside,degrees=True)
            
            for vv in ['ra','dec','z','sigmaz','m','sky_indices']:            
                cat.create_dataset(vv,data=cat_data[vv])
                
            cat.attrs['Ngal']=len(cat['z'])  
            
        self.hdf5pointer = h5py.File(filename,'r+')
        self.calc_kcorr=kcorr_dep(self.hdf5pointer['catalog'].attrs['band'])
    
    def load_hdf5(self,filename,cosmo_ref=None,epsilon=None):
        '''
        Loads the catalog HDF5 file
        
        Parameters
        ----------
        filename: string
            Name of the catalgo HDF5 file to load.
        cosmo_ref: class 
            Cosmology class used to create the catalog
        epsilon: float
            Luminosity weight index used to create the galaxy density interpolant
        '''
        
        self.hdf5pointer = h5py.File(filename,'r')
        self.sch_fun=galaxy_MF_dep(band=self.hdf5pointer['catalog'].attrs['band'])
        self.calc_kcorr=kcorr_dep(self.hdf5pointer['catalog'].attrs['band'])
        # Stores it internally
        try:
            if self.hdf5pointer['catalog/mthr_map'].attrs['mthr_percentile'] == 'empty':
                self.mthr_map_cpu = 'empty'
                if is_there_cupy:
                    self.mthr_map_gpu = 'empty'
            else:
                self.mthr_map_cpu = self.hdf5pointer['catalog/mthr_map/mthr_sky'][:]
                if is_there_cupy:
                    self.mthr_map_gpu = np2cp(self.hdf5pointer['catalog/mthr_map/mthr_sky'][:])
                
            print('Loading apparent magnitude threshold map')
        except:
            print('Apparent magnitude threshold not present')

        if cosmo_ref is not None:
            self.sch_fun.build_MF(cosmo_ref)
        
        if epsilon is not None:
            self.sch_fun.build_effective_number_density_interpolant(epsilon)
            
        try:
            if cosmo_ref is None:
                raise ValueError('You need to provide a cosmology if you want to load the interpolant')
            
            self.sch_fun.build_effective_number_density_interpolant(
                self.hdf5pointer['catalog/dNgal_dzdOm_interpolant'].attrs['epsilon'])
            interpogroup = self.hdf5pointer['catalog/dNgal_dzdOm_interpolant']
            
            dNgal_dzdOm_vals = []
            for i in range(self.hdf5pointer['catalog'].attrs['npixels']):
                dNgal_dzdOm_vals.append(interpogroup['vals_pixel_{:d}'.format(i)][:])

            self.dNgal_dzdOm_vals_cpu = np.column_stack(dNgal_dzdOm_vals)
            self.dNgal_dzdOm_vals_cpu[np.isnan(self.dNgal_dzdOm_vals_cpu)] = -np.inf
            self.dNgal_dzdOm_vals_cpu = np.exp(self.dNgal_dzdOm_vals_cpu.astype(float))
            
            self.dNgal_dzdOm_sky_mean_cpu = np.mean(self.dNgal_dzdOm_vals_cpu,axis=1)
            self.z_grid_cpu = interpogroup['z_grid'][:]
            self.pixel_grid_cpu = np.arange(0,self.hdf5pointer['catalog'].attrs['npixels'],1).astype(int)
            
            print('Loading Galaxy density interpolant')
            
            if is_there_cupy:
                self.dNgal_dzdOm_vals_gpu=np2cp(self.dNgal_dzdOm_vals_cpu)
                self.dNgal_dzdOm_sky_mean_gpu=np2cp(self.dNgal_dzdOm_sky_mean_cpu)
                self.z_grid_gpu = np2cp(self.z_grid_cpu)
                self.pixel_grid_gpu=np2cp(self.pixel_grid_cpu)
            
        except:
            print('interpolant not loaded')
    
    def calculate_mthr(self,mthr_percentile=50,nside_mthr=None):
        '''
        Calculates the apparent magnitude threshold as a function of the sky pixel.
        The apparent magnitude threshold is defined from the inverse CDF of galaxies reported in each pixel.
        
        Parameters
        ----------
        mthr_percentile: float
            Percentile to use to calculate the apparent magnitude threshold
        nside_mthr: int 
            Nside to compute threshold, it should be higher or equal than the one used to pixelise galaxies.
        '''
        
        if nside_mthr is None:
            nside_mthr = int(self.hdf5pointer['catalog'].attrs['nside'])
        skypixmthr = radec2indeces(self.hdf5pointer['catalog/ra'][:],self.hdf5pointer['catalog/dec'][:],nside_mthr)
        npixelsmthr = hp.nside2npix(nside_mthr)
        
        try:
            mthgroup = self.hdf5pointer['catalog'].create_group('mthr_map')
            mthgroup.attrs['mthr_percentile'] = mthr_percentile
            mthgroup.attrs['nside_mthr'] = nside_mthr
            mthgroup.create_dataset('mthr_sky',data=np.nan_to_num(
                -np.ones(self.hdf5pointer['catalog'].attrs['npixels'])*np.inf))
            indx_sky = 0 # An arra
        except:
            mthgroup = self.hdf5pointer['catalog/mthr_map']
            indx_sky = mthgroup.attrs['sky_checkpoint']
            print('Group already exists, resuming from pixel {:d}'.format(indx_sky))
        
        if mthr_percentile !='empty':
            # The block below computes the apparent magnitude threshold
            skyloop=np.arange(indx_sky,self.hdf5pointer['catalog'].attrs['npixels'],1).astype(int)

            for indx in tqdm(skyloop,desc='Calculating mthr in pixels'):
                mthgroup.attrs['sky_checkpoint']=indx
                rap, decp = indices2radec(indx,self.hdf5pointer['catalog'].attrs['nside'])
                bigpix = radec2indeces(rap,decp,nside_mthr)               
                ind=np.where(skypixmthr==bigpix)[0]
                if ind.size==0:
                    continue
                mthgroup['mthr_sky'][indx] = np.percentile(self.hdf5pointer['catalog/m'][ind],
                                                           mthgroup.attrs['mthr_percentile'])


            # The block below throws away all the galaxies fainter than
            # the apparent magnitude threshold
            castmthr=mthgroup['mthr_sky'][:][self.hdf5pointer['catalog/sky_indices'][:]]
            tokeep=np.where(self.hdf5pointer['catalog/m'][:]<=castmthr)[0]
            for vv in ['ra','dec','z','sigmaz','m','sky_indices']:
                tosave=self.hdf5pointer['catalog'][vv][:][tokeep]
                del self.hdf5pointer['catalog'][vv]       
                self.hdf5pointer['catalog'].create_dataset(vv,data=tosave)

            self.hdf5pointer['catalog'].attrs['Ngal']=len(tokeep)
            # Stores it internally
            self.mthr_map_cpu = self.hdf5pointer['catalog/mthr_map/mthr_sky'][:]
            if is_there_cupy:
                self.mthr_map_gpu=np2cp(self.mthr_map_cpu)
        else:
            # Set the counter to the last loop point
            mthgroup.attrs['sky_checkpoint']=self.hdf5pointer['catalog'].attrs['npixels']-1
            self.mthr_map_cpu='empty'
            if is_there_cupy:
                self.mthr_map_gpu='empty'
    
    def return_counts_map(self):
        '''
        Returns the galaxy counts in the skymap as np.array
        '''
        npixels = self.hdf5pointer['catalog'].attrs['Ngal']
        counts_map = np.zeros(npixels)
        for indx in range(npixels):
            ind=np.where(self.hdf5pointer['catalog/sky_indices'][:]==indx)[0]
            counts_map[indx]=len(ind)
            if ind.size==0:
                continue
        return counts_map
                
    def plot_mthr_map(self,**kwargs):
        '''
        Plots the mthr_map. Use **kwargs parameters for the hp.mollview
        '''
        mtr_map = self.hdf5pointer['catalog/mthr_map/mthr_sky'][:]
        mtr_map[mtr_map==LOWERL]=hp.UNSEEN
        mtr_map=hp.ma(mtr_map)
        hp.mollview(mtr_map,**kwargs)

    def plot_counts_map(self,**kwargs):
        '''
        Plots galaxy counts map. Use **kwargs parameters for the hp.mollview
        '''
        count=self.return_counts_map()
        count[count==0]=hp.UNSEEN
        count=hp.ma(count)
        hp.mollview(count,**kwargs)
        
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
        
        
        if dl is None:
            dl=cosmology.z2dl(z)
        
        if iscupy(z):
            mthr_map = self.mthr_map_gpu
        else:
            mthr_map = self.mthr_map_cpu
        
        mthr_arrays=mthr_map[radec_indices]
        return m2M(mthr_arrays,dl,self.calc_kcorr(z))
    
        
    def calc_dN_by_dzdOmega_interpolant(self,cosmo_ref,epsilon,
                                        Nintegration=10,Numsigma=1,
                                        zcut=None,ptype='uniform'):
        '''
        Fits the dNgal/dzdOmega interpolant
        
        Parameters
        ----------
        cosmo_ref: class 
            Cosmology used to compute the differential of comoving volume (normalized)
        epsilon: float
            Luminosity weight
        Nres: int 
            Increasing factor for the interpolation array in z
        Numsigma: float
            Half Width for the uniform distribution method in terms of sigmaz
        zcut: float
            Redshift where to cut the galaxy catalog, after zcut the completeness is 0
        ptype: string
            'uniform' or 'gaussian' for the EM likelihood type of galaxies
        '''
        
        self.sch_fun=galaxy_MF_depr(band=self.hdf5pointer['catalog'].attrs['band'])
        self.sch_fun.build_effective_number_density_interpolant(epsilon)

        # If zcut is none, it uses the maximum of the cosmology
        if zcut is None:
            zcut = cosmo_ref.zmax
        
        # Overrides the num of sigma for the gaussian
        if (ptype == 'gaussian') | (ptype == 'gaussian_nocom'):
            print('Setting 5 sigma for the gaussian normalization')
            Numsigma=5.
            
        try:
            interpogroup = self.hdf5pointer['catalog'].create_group('dNgal_dzdOm_interpolant')
            interpogroup.attrs['epsilon'] = epsilon
            interpogroup.attrs['ptype']=ptype
            interpogroup.attrs['Nintegration']=Nintegration
            interpogroup.attrs['Numsigma']=Numsigma
            interpogroup.attrs['zcut']=zcut
            indx_sky = 0 # An arra
        except:
            interpogroup = self.hdf5pointer['catalog/dNgal_dzdOm_interpolant']
            indx_sky = interpogroup.attrs['sky_checkpoint']
            print('Group already exists, resuming from pixel {:d}'.format(indx_sky))
        
        self.sch_fun.build_MF(cosmo_ref)
        
        cat_data=self.hdf5pointer['catalog']
        
        # If zcut is none, it uses the maximum of the cosmology
        if zcut is None:
            zcut = cosmo_ref.zmax
        
        # Selects all the galaxies that have support below zcut and above 1e-6
        idx_in_range = np.where(((cat_data['z'][:]-Numsigma*cat_data['sigmaz'][:])<=zcut) & ((cat_data['z'][:]+Numsigma*cat_data['sigmaz'][:])>=1e-6))[0]
        if len(idx_in_range)==0:
            raise ValueError('There are no galaxies in the redshift range 1e-6 - {:f}'.format(zmax))
                
        interpolation_width = np.empty(len(idx_in_range),dtype=np.float32)
        j = 0
        for i in tqdm(idx_in_range,desc='Looping on galaxies to find width'):
            zmin = np.max([cat_data['z'][i]-Numsigma*cat_data['sigmaz'][i],1e-6])
            zmax = np.min([cat_data['z'][i]+Numsigma*cat_data['sigmaz'][i],zcut])
            if zmax>=cosmo_ref.zmax:
                print(zmin,zmax)
                raise ValueError('The maximum redshift for interpolation is too high w.r.t the cosmology class')        
            interpolation_width[j]=zmax-zmin
            j+=1
            
        idx_sorted = np.argsort(interpolation_width)
        del interpolation_width
        idx_sorted = idx_sorted[::-1] # Decreasing order
        
        z_grid = np.linspace(1e-6,zcut,Nintegration)
        # Note that idx_in_range[idx_sorted] is the label of galaxies such that the 
        # interpolation width is sorted in decreasing order
        for i in tqdm(idx_in_range[idx_sorted],desc='Looping galaxies to find array'):
            zmin = np.max([cat_data['z'][i]-Numsigma*cat_data['sigmaz'][i],1e-6])
            zmax = np.min([cat_data['z'][i]+Numsigma*cat_data['sigmaz'][i],zcut])
            zinterpolator = np.linspace(zmin,zmax,Nintegration)
            delta=(zmax-zmin)/Nintegration
            z_grid = np.sort(np.hstack([z_grid,zinterpolator]))
            even_in = np.arange(0,len(z_grid),2)
            odd_in = np.arange(1,len(z_grid),2)
            z_even = z_grid[even_in]
            diffv = np.diff(z_even)
            to_eliminate = odd_in[np.where(diffv<delta)[0]]
            z_grid = np.delete(z_grid,to_eliminate)
            
        z_grid = np.unique(z_grid)
             
        absM_rate=log_powerlaw_absM_rate(epsilon=epsilon)
        print('Z array is long {:d}'.format(len(z_grid)))

        if indx_sky == 0:
            interpogroup.create_dataset('z_grid', data = z_grid)
            interpogroup.create_dataset('pixel_grid', data = np.arange(0,self.hdf5pointer['catalog'].attrs['npixels'],1).astype(int))
        
        skyloop=np.arange(indx_sky,self.hdf5pointer['catalog'].attrs['npixels'],1).astype(int)
        cpind=cat_data['sky_indices'][:][idx_in_range]    
        
        for i in tqdm(skyloop,desc='Calculating interpolant'):
            interpogroup.attrs['sky_checkpoint']=i
            gal_index=np.where(cpind==i)[0]
            if len(gal_index)==0:
                tos = np.zeros_like(z_grid)
                tos[:] = np.nan
                interpogroup.create_dataset('vals_pixel_{:d}'.format(i), data = tos,dtype = np.float16)
                del tos
                continue
            
            interpo = 0.
            
            for gal in gal_index:
                # List of galaxy catalog density in increasing order per pixel. This corresponds to Eq. 2.35 on the overleaf document
                Mv=m2M(cat_data['m'][idx_in_range[gal]],cosmo_ref.z2dl(z_grid),self.calc_kcorr(z_grid))               
                interpo+=absM_rate.evaluate(self.sch_fun,Mv)*EM_likelihood_prior_differential_volume(z_grid,
                                                            cat_data['z'][idx_in_range[gal]],cat_data['sigmaz'][idx_in_range[gal]],cosmo_ref
                                                            ,Numsigma=Numsigma,ptype=ptype)/self.hdf5pointer['catalog'].attrs['dOmega_sterad']
            
            interpo[interpo==0.]=np.nan                
            interpo = np.float16(np.log(interpo))
            interpogroup.create_dataset('vals_pixel_{:d}'.format(i), data = interpo, dtype = np.float16)
        
        self.hdf5pointer.close()
                
    def effective_galaxy_number_interpolant(self,z,skypos,cosmology,dl=None,average=False):
        '''
        Returns an evaluation of dNgal/dzdOmega, it requires `calc_dN_by_dzdOmega_interpolant` to be called first.
        
        Parameters
        ----------
        z: xp.array
            Redshift array
        skypos: xp.array
            Array containing the healpix indeces where to evaluate the interpolant
        cosmology: class
            cosmology class to use for the computation
        dl: xp.array
            Luminosity distance in Mpc
        average: bool
            Use the sky averaged differential of effective number of galaxies in each pixel
        '''
        xp=get_module_array(z)
        sx=get_module_array_scipy(z)
       
        originshape=z.shape
        z=z.flatten()
        self.sch_fun.build_MF(cosmology)
        skypos=skypos.flatten()
        
        if dl is None:
            dl=cosmology.z2dl(z)
        dl=dl.flatten()
        
        if isinstance(self.mthr_map_cpu, str):
            return xp.zeros(len(z)).reshape(originshape), (self.sch_fun.background_effective_galaxy_density(-xp.inf*xp.ones_like(z))*cosmology.dVc_by_dzdOmega_at_z(z)).reshape(originshape)
        
        if iscupy(z):
            z_grid = self.z_grid_gpu
            dNgal_dzdOm_sky_mean = self.dNgal_dzdOm_sky_mean_gpu
            dNgal_dzdOm_vals = self.dNgal_dzdOm_vals_gpu
            pixel_grid = self.pixel_grid_gpu
        else:
            z_grid = self.z_grid_cpu
            dNgal_dzdOm_sky_mean = self.dNgal_dzdOm_sky_mean_cpu
            dNgal_dzdOm_vals = self.dNgal_dzdOm_vals_cpu
            pixel_grid = self.pixel_grid_cpu
        
        Mthr_array=self.calc_Mthr(z,skypos,cosmology,dl=dl)
        # Baiscally tells that if you are above the maximum interpolation range, you detect nothing
        Mthr_array[z>z_grid[-1]]=-xp.inf
        gcpart,bgpart=xp.zeros_like(z),xp.zeros_like(z)
        
        if average:
            gcpart=xp.interp(z,z_grid,dNgal_dzdOm_sky_mean,left=0.,right=0.)
        else:
            gcpart=sx.interpolate.interpn((z_grid,pixel_grid),dNgal_dzdOm_vals,xp.column_stack([z,skypos]),bounds_error=False,
                                fill_value=0.,method='linear') # If a posterior samples fall outside, then you return 0
        
        bgpart=self.sch_fun.background_effective_galaxy_density(Mthr_array)*cosmology.dVc_by_dzdOmega_at_z(z)
        
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
            Mthr_array[z>self.z_grid_cpu[-1]]=-np.inf
            inco[:,i]=self.sch_fun.background_effective_galaxy_density(Mthr_array)/self.sch_fun.background_effective_galaxy_density(-np.ones_like(Mthr_array)*np.inf)
            
        fig,ax=plt.subplots(2,1,sharex=True)
        
        theo=self.sch_fun.background_effective_galaxy_density(-np.inf*np.ones_like(z))*cosmology.dVc_by_dzdOmega_at_z(z)
                
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


# LVK Reviewed    
class galaxy_MF_dep(object):
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
class kcorr_dep(object):
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

    
