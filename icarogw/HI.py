from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn, is_there_cupy

class HI_map(object):
    def __init__(self,redshift_grid, pixel_grid, density_matrix):
        '''
        Ciao
        Parameters
        ----------
        redshift_grid: xp.array
            Redshift grid used to construct the density matrix
        pixel_grid: xp.array
            Pixel grid (healpy or mhealpy) indeces to construct the matrix
        density_matrix: xp.array
            Density Matrix 
        '''
        xp=get_module_array(redshift_grid)
        self.redshift_grid = redshift_grid
        self.pixel_grid = pixel_grid
        self.density_matrix = density_matrix   
        self.density_matrix_average = xp.mean(density_matrix,axis=1) # Check axis 	
    
    def drho_dzdomega(self,z,skypos,cosmology,dl=None,average=False):
        '''
        Parameters
        ----------
        z: xp.array
            Redshift array
        skypos: xp.array
            Array containing the healpix indeces where to evaluate the interpolant (same indexing as grid interpolant)
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
        skypos=skypos.flatten()
        
        if dl is None:
            dl=cosmology.z2dl(z)
        dl=dl.flatten()
        
        z_grid = self.redshift_grid
        dNgal_dzdOm_sky_mean = self.density_matrix_average
        dNgal_dzdOm_vals = self.density_matrix
        pixel_grid = self.pixel_grid
                
        if average:
            gcpart=xp.interp(z,z_grid,dNgal_dzdOm_sky_mean,left=0.,right=0.)
        else:
            gcpart=sx.interpolate.interpn((z_grid,pixel_grid),dNgal_dzdOm_vals,xp.column_stack([z,skypos]),bounds_error=False,
                                fill_value=0.,method='linear') # If a posterior samples fall outside, then you return 0
        
        return gcpart.reshape(originshape)
