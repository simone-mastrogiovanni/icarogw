from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn, is_there_cupy
from .conversions import radec2indeces, indices2radec, M2m, m2M
from .cosmology import galaxy_MF, kcorr, log_powerlaw_absM_rate

import healpy as hp
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

LOWERL=np.nan_to_num(-np.inf)

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
        self.calc_kcorr=kcorr(self.hdf5pointer['catalog'].attrs['band'])
    
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
        self.sch_fun=galaxy_MF(band=self.hdf5pointer['catalog'].attrs['band'])
        self.calc_kcorr=kcorr(self.hdf5pointer['catalog'].attrs['band'])
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
        
        self.sch_fun=galaxy_MF(band=self.hdf5pointer['catalog'].attrs['band'])
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
        
 

    
    
    
    
    
    
    
