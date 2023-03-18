from .cupy_pal import *
import copy as cp
from .conversions import  radec2indeces

class injections(object):
    def __init__(self,injections_dict,prior,ntotal,Tobs):

        '''
        This class is used to manage a list of detected injections to calculate
        GW selection effects. This class uses injections which are given in source frame.
        injections_dict: xp.array
            Dictionary containing the variables with which you want to evaluate the injections.
        prior: xp.array
            Used prior draws for injections, same as the ones in injections dict p(mass1,mass2,dl,ra,dec)
        ntotal: float
            Total number of simulated injections (detected and not). This is necessary to compute the expected number of detections
        Tobs: float
            Length of time for the run in years (used to calculate rates)
        '''
        
        # Saves what you provided in the class
        self.injections_data_original={key:injections_dict[key] for key in injections_dict.keys()}
        self.injections_data={key:injections_dict[key] for key in injections_dict.keys()}
        self.prior_original=cp.deepcopy(prior)
        self.prior=cp.deepcopy(prior)
        self.detection_index=xp.ones_like(prior,dtype=bool)
        self.ntotal=ntotal
        self.Tobs=Tobs
        
    def update_cut(self,detection_index):
        ''' Update the selection cut and the injections that you are able to detect
        
        Parameters
        ----------
        detection_index: xp.array
            Array with True where you detect the injection and False otherwise
        '''
        self.detection_index=detection_index
        self.injections_data={key:self.injections_data_original[key][detection_index] for key in self.injections_data_original.keys()}
        self.prior=self.prior_original[detection_index]
        
    def cupyfy(self):
        ''' Converts all the posterior samples to cupy'''
        self.injections_data_original={key:np2cp(self.injections_data_original[key]) for key in self.injections_data_original.keys()}
        self.injections_data={key:np2cp(self.injections_data[key]) for key in self.injections_data_original.keys()}
        self.prior_original=np2cp(self.prior_original)
        self.prior=np2cp(self.prior)
        
    def numpyfy(self):
        ''' Converts all the posterior samples to numpy'''
        self.injections_data_original={key:cp2np(self.injections_data_original[key]) for key in self.injections_data_original.keys()}
        self.injections_data={key:cp2np(self.injections_data[key]) for key in self.injections_data_original.keys()}
        self.prior_original=cp2np(self.prior_original)
        self.prior=cp2np(self.prior)
        
    def effective_detection_number(self,weights):
        mean = xp.sum(weights)/self.ntotal
        var = xp.sum(weights**2)/(self.ntotal**2)-(mean**2)/self.ntotal
        return (mean**2)/var
    
    def pixelize(self,nside):
        self.nside=nside
        self.injections_data_original['sky_indices'] = radec2indeces(
            cp2np(self.injections_data_original['right_ascension']),
            cp2np(self.injections_data_original['declination']),nside)
        self.injections_data['sky_indices'] = radec2indeces(
            cp2np(self.injections_data['right_ascension']),
            cp2np(self.injections_data['declination']),nside)
        
        if isinstance(self.prior,np.ndarray):
            self.numpyfy()
            
    def update_weights(self,rate_wrapper):
        self.weights = xp.exp(rate_wrapper.log_rate_injections(self.prior,**{key:self.injections_data[key] for key in self.injections_data.keys()}))
        self.pseudo_rate = xp.sum(self.weights)/self.ntotal
        
    def expected_number_detections(self):
        return self.Tobs*self.pseudo_rate
 