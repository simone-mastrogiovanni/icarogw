from .cupy_pal import *
from .conversions import radec2indeces

class posterior_samples(object):
    def __init__(self,posterior_dict,prior):
        '''
        Class to handle posterior samples for icarogwCAT.
        
        Parameters
        ----------
        posterior_dict: np.array
            Dictionary of posterior samples
        prior: np.array
            Prior to use in order to reweight posterior samples written in the same variables that you provide, e.g. if you provide d_l and m1d, then p(d_l,m1d)
        '''
        self.posterior_data={key: posterior_dict[key] for key in posterior_dict.keys()}
        self.posterior_data['prior']=prior
        self.nsamples=len(prior)
        
    def pixelize(self,nside):
        '''
        Pixelize the Ra and DEC samples
        
        Parameters
        ----------
        nside: integer
            Nside for healpy
        '''
        self.posterior_data['sky_indices'] = radec2indeces(self.posterior_data['right_ascension'],self.posterior_data['declination'],nside)
        self.nside=nside
        
    def cupyfy(self):
        ''' Converts all the posterior samples to cupy'''
        self.posterior_data={key:np2cp(self.posterior_data[key]) for key in self.posterior_data}
        
    def numpyfy(self):
        ''' Converts all the posterior samples to numpy'''
        self.posterior_data={key:cp2np(self.posterior_data[key]) for key in self.posterior_data}
        
    def add_counterpart(self,z_EM,ra,dec):
        
        idx = radec2indeces(ra,dec,self.nside)
        select = xp.where(self.posterior_data['sky_indices']==idx)[0]
        print('There are {:d} samples in the EM counterpart direction'.format(len(select)))
        self.posterior_data={key: self.posterior_data[key] for key in self.posterior_data.keys()}
        self.posterior_data['z_EM'] = z_EM
        self.nsamples=len(self.posterior_data['sky_indices'])
        
        
