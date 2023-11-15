from .cupy_pal import cp2np, np2cp, get_module_array, get_module_array_scipy, iscupy, np, sn, check_bounds_1D, check_bounds_2D
from .conversions import L2M, M2L
import copy

# LVK Reviewed
def betadistro_muvar2ab(mu,var):
    '''
    Calculates the a and b parameters of the beta distribution given mean and variance
    
    Parameters:
    -----------
    mu, var: xp.array
        mean and variance of the beta distribution
    
    Returns:
    --------
    a,b: xp.array 
        a and b parameters of the beta distribution
    '''
    
    xp = get_module_array(mu)
    
    a=(((1.-mu)/var)- 1./mu )*xp.power(mu,2.)
    return a,a* (1./mu -1.)

# LVK Reviewed
def betadistro_ab2muvar(a,b):
    '''
    Calculates the a and b parameters of the beta distribution given mean and variance
    
    Parameters:
    -----------
    a,b: xp.array 
        a and b parameters of the beta distribution
    
    Returns:
    --------
    mu, var: xp.array
        mean and variance of the beta distribution
    '''
    xp = get_module_array(a)
    
    return a/(a+b), a*b/(xp.power(a+b,2.) *(a+b+1.))

# LVK Reviewed
def _S_factor(mass, mmin,delta_m):
    '''
    This function returns the value of the window function defined as Eqs B6 and B7 of https://arxiv.org/pdf/2010.14533.pdf

    Parameters
    ----------
    mass: xp.array or float
        array of x or masses values
    mmin: float or xp.array (in this case len(mmin) == len(mass))
        minimum value of window function
    delta_m: float or xp.array (in this case len(delta_m) == len(mass))
        width of the window function

    Returns
    -------
    Values of the window function
    '''
    
    xp = get_module_array(mass)

    to_ret = xp.ones_like(mass)
    if delta_m == 0:
        return to_ret

    mprime = mass-mmin

    # Defines the different regions of thw window function ad in Eq. B6 of  https://arxiv.org/pdf/2010.14533.pdf
    select_window = (mass>mmin) & (mass<(delta_m+mmin))
    select_one = mass>=(delta_m+mmin)
    select_zero = mass<=mmin

    effe_prime = xp.ones_like(mass)

    # Defines the f function as in Eq. B7 of https://arxiv.org/pdf/2010.14533.pdf
    # This line might raise a warnig for exp orverflow, however this is not important as it enters at denominator
    effe_prime[select_window] = xp.exp(xp.nan_to_num((delta_m/mprime[select_window])+(delta_m/(mprime[select_window]-delta_m))))
    to_ret = 1./(effe_prime+1)
    to_ret[select_zero]=0.
    to_ret[select_one]=1.
    return to_ret

# LVK Reviewed
class basic_1dimpdf(object):
    
    def __init__(self,minval,maxval):
        '''
        Basic class for a 1-dimensional pdf
        
        Parameters
        ----------
        minval,maxval: float
            minimum and maximum values within which the pdf is defined
        '''
        self.minval=minval
        self.maxval=maxval 
        
    def _check_bound_pdf(self,x,y):
        '''
        Check if x is between the pdf boundaries and set y to -xp.inf where x is outside
        
        Parameters
        ----------
        x,y: xp.array
            Array where the log pdf is evaluated and values of the log pdf
        
        Returns
        -------
        log pdf values: xp.array
            log pdf values updates to -xp.inf outside the boundaries
            
        '''
        xp = get_module_array(x)
        indx=check_bounds_1D(x,self.minval,self.maxval)
        y[indx]=-xp.inf
        return y
    
    def _check_bound_cdf(self,x,y):
        '''
        Check if x is between the pdf boundaries nd set the cdf y to 0 and 1 outside the boundaries
        
        Parameters
        ----------
        x,y: xp.array
            Array where the log cdf is evaluated and values of the log cdf
        
        Returns
        -------
        log cdf values: xp.array
            log cdf values updates to 0 and 1 outside the boundaries
            
        '''
        xp = get_module_array(x)
        y[x<self.minval],y[x>self.maxval]=-xp.inf,0.
        return y
    
    def log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        y=self._log_pdf(x)
        y=self._check_bound_pdf(x,y)
        return y
    
    def log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: xp.array
        '''
        y=self._log_cdf(x)
        y=self._check_bound_cdf(x,y)
        return y
    
    def pdf(self,x):
        '''
        Evaluates the pdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the pdf
        
        Returns
        -------
        pdf: xp.array
        '''
        xp = get_module_array(x)
        return xp.exp(self.log_pdf(x))
    
    def cdf(self,x):
        '''
        Evaluates the cdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the cdf
        
        Returns
        -------
        cdf: xp.array
        '''
        xp = get_module_array(x)
        return xp.exp(self.log_cdf(x))
    
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
        sarray=np.linspace(self.minval,self.maxval,10000)
        cdfeval=self.cdf(sarray)
        randomcdf=np.random.rand(N)
        return np.interp(randomcdf,cdfeval,sarray)

class paired_2dimpdf(object):
    
    def __init__(self,pdf,pairing_function):
        '''
        Class for a pairing mass function
        
        Parameters
        ----------
        pdf1: first pdf
        pairing function: python function that pairs m1 and m2
        '''
        self.pdf_base=pdf
        self.pairing_function=pairing_function
        self.norm = self._get_norm_factor()

    def _get_norm_factor(self):

        m1 = self.pdf_base.sample(10000)
        m2 = self.pdf_base.sample(10000)
        xp = get_module_array(m1)
        self.new_norm = xp.mean(self.pairing_function(m1,m2))

    def log_pdf(self,x1,x2):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x1,x2: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        # This line might create some nan since p(m2|m1) = p(m2)/CDF_m2(m1) = 0/0 if m2 and m1 < mmin.
        # This nan is eliminated with the _check_bound_pdf
        xp = get_module_array(x1)
        y=self.pdf_base.log_pdf(x1)+self.pdf_base.log_pdf(x2)+xp.log(self.pairing_function(x1,x2))-xp.log(self.new_norm)
        y[xp.isnan(y)]=-xp.inf
        return y 
    
    def pdf(self,x1,x2):
        '''
        Evaluates the pdf
        
        Parameters
        ----------
        x1,x2: xp.array
            where to evaluate the pdf
        
        Returns
        -------
        pdf: xp.array
        '''
        xp = get_module_array(x1)
        return xp.exp(self.log_pdf(x1,x2))

# LVK Reviewed
class conditional_2dimpdf(object):
    
    def __init__(self,pdf1,pdf2):
        '''
        Basic class for a 2-dimensional pdf, where x2<x1
        
        Parameters
        ----------
        pdf1, pdf2: basic_1dimpdf, basic_2dimpdf
            Two classes of pdf functions
        '''
        self.pdf1=pdf1
        self.pdf2=pdf2
        
    def _check_bound_pdf(self,x1,x2,y):
        '''
        Check if x1 and x2 are between the pdf boundaries and set y to -xp.inf where x1<x2 is outside
        
        Parameters
        ----------
        x1,x2,y: xp.array
            Array where the log pdf is evaluated and values of the log pdf
        
        Returns
        -------
        log pdf values: xp.array
            log pdf values updated to -xp.inf outside the boundaries
            
        '''
        xp = get_module_array(x1)
        indx=check_bounds_2D(x1,x2,y)
        y[indx]=-xp.inf
        return y
    
    def log_pdf(self,x1,x2):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x1,x2: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        # This line might create some nan since p(m2|m1) = p(m2)/CDF_m2(m1) = 0/0 if m2 and m1 < mmin.
        # This nan is eliminated with the _check_bound_pdf
        y=self.pdf1.log_pdf(x1)+self.pdf2.log_pdf(x2)-self.pdf2.log_cdf(x1)
        y=self._check_bound_pdf(x1,x2,y)
        return y 
    
    def pdf(self,x1,x2):
        '''
        Evaluates the pdf
        
        Parameters
        ----------
        x1,x2: xp.array
            where to evaluate the pdf
        
        Returns
        -------
        pdf: xp.array
        '''
        xp = get_module_array(x1)
        return xp.exp(self.log_pdf(x1,x2))
    
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
        sarray1=np.linspace(self.pdf1.minval,self.pdf1.maxval,10000)
        cdfeval1=self.pdf1.cdf(sarray1)
        randomcdf1=np.random.rand(N)

        sarray2=np.linspace(self.pdf2.minval,self.pdf2.maxval,10000)
        cdfeval2=self.pdf2.cdf(sarray2)
        randomcdf2=np.random.rand(N)
        x1samp=np.interp(randomcdf1,cdfeval1,sarray1)
        x2samp=np.interp(randomcdf2*self.pdf2.cdf(x1samp),cdfeval2,sarray2)
        return x1samp,x2samp

# LVK Reviewed
class SmoothedProb(basic_1dimpdf):
    
    def __init__(self,originprob,bottomsmooth):
        '''
        Class for a smoother probability
        
        Parameters
        ----------
       
        '''
        self.origin_prob = copy.deepcopy(originprob)
        self.bottom_smooth = bottomsmooth
        self.bottom = originprob.minval
        super().__init__(originprob.minval,originprob.maxval)
        
        # Find the values of the integrals in the region of the window function before and after the smoothing
        int_array = np.linspace(originprob.minval,originprob.minval+bottomsmooth,1000)
        integral_before = np.trapz(self.origin_prob.pdf(int_array),int_array)
        integral_now = np.trapz(self.origin_prob.pdf(int_array)*_S_factor(int_array, self.bottom,self.bottom_smooth),int_array)

        self.integral_before = integral_before
        self.integral_now = integral_now
        # Renormalize the smoother function.
        self.norm = 1 - integral_before + integral_now

        self.x_eval_cpu = np.linspace(self.bottom,self.bottom+self.bottom_smooth,1000)
        self.cdf_numeric_cpu = np.cumsum(self.pdf((self.x_eval_cpu[:-1:]+self.x_eval_cpu[1::])*0.5))*(self.x_eval_cpu[1::]-self.x_eval_cpu[:-1:])
        
        self.x_eval_gpu = np2cp(self.x_eval_cpu)
        self.cdf_numeric_gpu = np2cp(self.cdf_numeric_cpu)
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        xp = get_module_array(x)
        # Return the window function
        window = _S_factor(x, self.bottom,self.bottom_smooth)
        # The line below might raise warnings for log(0), however python is able to handle it.
        prob_ret =self.origin_prob.log_pdf(x)+xp.log(window)-xp.log(self.norm)
        return prob_ret

    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: xp.array
        '''
        xp = get_module_array(x)
        
        if iscupy(x):
            cdf_numeric = self.cdf_numeric_gpu
            x_eval = self.x_eval_gpu
        else:
            cdf_numeric = self.cdf_numeric_cpu
            x_eval = self.x_eval_cpu
        
        origin=x.shape
        ravelled=xp.ravel(x)
        
        toret = xp.ones_like(ravelled)
        toret[ravelled<self.bottom] = 0.        
        toret[(ravelled>=self.bottom) & (ravelled<=(self.bottom+self.bottom_smooth))] = xp.interp(ravelled[(ravelled>=self.bottom) & (ravelled<=(self.bottom+self.bottom_smooth))]
                           ,(x_eval[:-1:]+x_eval[1::])*0.5,cdf_numeric)
        # The line below might contain some log 0, which is automatically accounted for in python
        toret[ravelled>=(self.bottom+self.bottom_smooth)]=(self.integral_now+self.origin_prob.cdf(
        ravelled[ravelled>=(self.bottom+self.bottom_smooth)])-self.origin_prob.cdf(xp.array([self.bottom+self.bottom_smooth])))/self.norm
        
        return xp.log(toret).reshape(origin)

# LVK Reviewed
def PL_normfact(minpl,maxpl,alpha):
    '''
    Returns the Powerlaw normalization factor
    
    Parameters
    ----------
    minpl, maxpl,alpha: Minimum, maximum and power law exponent of the distribution
    '''
    if alpha == -1:
        norm_fact=np.log(maxpl/minpl)
    else:
        norm_fact=(np.power(maxpl,alpha+1.)-np.power(minpl,alpha+1))/(alpha+1)
    return norm_fact

# LVK Reviewed
class PowerLaw(basic_1dimpdf):
    
    def __init__(self,minpl,maxpl,alpha):
        '''
        Class for a  powerlaw probability
        
        Parameters
        ----------
        minpl,maxpl,alpha: float
            Minimum, Maximum and exponent of the powerlaw 
        '''
        super().__init__(minpl,maxpl)
        self.minpl,self.maxpl,self.alpha=minpl, maxpl, alpha
        self.norm_fact=PL_normfact(minpl,maxpl,alpha)
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        xp = get_module_array(x)
        toret=self.alpha*xp.log(x)-xp.log(self.norm_fact)
        return toret
    
    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: xp.array
        '''
        xp = get_module_array(x)
        if self.alpha == -1.:
            toret = xp.log(xp.log(x/self.minval)/self.norm_fact)
        else:
            toret =xp.log(((xp.power(x,self.alpha+1)-xp.power(self.minpl,self.alpha+1))/(self.alpha+1))/self.norm_fact)
        return toret

# LVK Reviewed
def get_beta_norm(alpha, beta):
    ''' 
    This function returns the normalization factor of the Beta PDF
    
    Parameters
    ----------
    alpha: float s.t. alpha > 0
           first component of the Beta law
    beta: float s.t. beta > 0
          second component of the Beta law
    '''
    
    # Get the Beta norm as in Wiki Beta function https://en.wikipedia.org/wiki/Beta_distribution
    return sn.special.gamma(alpha)*sn.special.gamma(beta)/sn.special.gamma(alpha+beta)

# LVK Reviewed
class BetaDistribution(basic_1dimpdf):
    
    def __init__(self,alpha,beta):
        '''
        Class for a Beta distribution probability
        
        Parameters
        ----------
        minbeta,maxbeta: float
            Minimum, Maximum of the beta distribution, they must be in 0,1
        alpha, beta: Parameters for the beta distribution
        '''
        super().__init__(0.,1.)
        self.alpha, self.beta = alpha, beta
        # Get the norm  (as described in https://en.wikipedia.org/wiki/Beta_distribution)
        self.norm_fact = get_beta_norm(self.alpha, self.beta)
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        xp = get_module_array(x)
        toret=(self.alpha-1.)*xp.log(x)+(self.beta-1.)*xp.log1p(-x)-xp.log(self.norm_fact)
        return toret
    
    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: xp.array
        '''
        xp = get_module_array(x)
        sx = get_module_array_scipy(x)
        toret = xp.log(sx.special.betainc(self.alpha,self.beta,x))
        return toret
        
# LVK Reviewed       
class TruncatedBetaDistribution(basic_1dimpdf):
    
    def __init__(self,alpha,beta,maximum):
        '''
        Class for a Truncated Beta distribution probability
        
        Parameters
        ----------
        minbeta,maxbeta: float
            Minimum, Maximum of the beta distribution, they must be in 0,max
        alpha, beta: Parameters for the beta distribution
        '''
        super().__init__(0.,maximum)
        self.alpha, self.beta, self.maximum = alpha, beta, maximum
        # Get the norm  (as described in https://en.wikipedia.org/wiki/Beta_distribution)
        self.norm_fact = get_beta_norm(self.alpha, self.beta)*sn.special.betainc(self.alpha,self.beta,self.maximum)
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        xp = get_module_array(x)
        toret=(self.alpha-1.)*xp.log(x)+(self.beta-1.)*xp.log1p(-x)-xp.log(self.norm_fact)
        return toret
    
    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: xp.array
        '''
        xp = get_module_array(x)
        sx = get_module_array_scipy(x)
        toret = xp.log(sx.special.betainc(self.alpha,self.beta,x)/sx.special.betainc(self.alpha,self.beta,self.maximum))
        return toret
        
# LVK Reviewed
def get_gaussian_norm(ming,maxg,meang,sigmag):
    '''
    Returns the normalization of the gaussian distribution
    
    Parameters
    ----------
    ming, maxg, meang,sigmag: Minimum, maximum, mean and standard deviation of the gaussian distribution
    '''
    max_point = (maxg-meang)/(sigmag*np.sqrt(2.))
    min_point = (ming-meang)/(sigmag*np.sqrt(2.))
    return 0.5*sn.special.erf(max_point)-0.5*sn.special.erf(min_point)

# LVK Reviewed
class TruncatedGaussian(basic_1dimpdf):
    
    def __init__(self,meang,sigmag,ming,maxg):
        '''
        Class for a Truncated gaussian probability
        
        Parameters
        ----------
        meang,sigmag,ming,maxg: float
            mean, sigma, min value and max value for the gaussian
        '''
        super().__init__(ming,maxg)
        self.meang,self.sigmag,self.ming,self.maxg=meang,sigmag,ming,maxg
        self.norm_fact= get_gaussian_norm(ming,maxg,meang,sigmag)
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        xp = get_module_array(x)
        toret=-xp.log(self.sigmag)-0.5*xp.log(2*xp.pi)-0.5*xp.power((x-self.meang)/self.sigmag,2.)-xp.log(self.norm_fact)
        return toret
    
    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: xp.array
        '''
        xp = get_module_array(x)
        sx = get_module_array_scipy(x)
        max_point = (x-self.meang)/(self.sigmag*xp.sqrt(2.))
        min_point = (self.ming-self.meang)/(self.sigmag*xp.sqrt(2.))
        toret = xp.log((0.5*sx.special.erf(max_point)-0.5*sx.special.erf(min_point))/self.norm_fact)
        return toret

# Overwrite most of the methods of the parent class
# Idea from https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution
# LVK Reviewed
class Bivariate2DGaussian(conditional_2dimpdf):
    
    def __init__(self,x1min,x1max,x1mean,x2min,x2max,x2mean,x1variance,x12covariance,x2variance):
        '''
        Basic class for a 2-dimensional pdf, where x2<x1
        
        Parameters
        ----------
        '''
        
        self.x1min,self.x1max,self.x1mean,self.x2min,self.x2max,self.x2mean=x1min,x1max,x1mean,x2min,x2max,x2mean
        self.x1variance,self.x12covariance,self.x2variance=x1variance,x12covariance,x2variance
        self.norm_marginal_1=get_gaussian_norm(self.x1min,self.x1max,self.x1mean,self.x1variance**0.5)
        
    def _check_bound_pdf(self,x1,x2,y):
        '''
        Check if x1 and x2 are between the pdf boundaries nd set y to -xp.inf where x is outside
        
        Parameters
        ----------
        x1,x2,y: xp.array
            Arrays where the log pdf is evaluated and values of the log pdf
        
        Returns
        -------
        log pdf values: xp.array
            log pdf values updates to -xp.inf outside the boundaries
            
        '''
        xp = get_module_array(x1)
        y[(x1<self.x1min) | (x1>self.x1max) | (x2<self.x2min) | (x2>self.x2max)]=-xp.inf
        return y
    
    def log_pdf(self,x1,x2):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x1,x2: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array 
            formulas from https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case_2
        '''
        xp = get_module_array(x1)
        marginal_log=-0.5*xp.log(2*xp.pi*self.x1variance)-0.5*xp.power(x1-self.x1mean,2.)/self.x1variance-xp.log(self.norm_marginal_1)
        
        conditioned_mean=self.x2mean+(self.x12covariance/self.x1variance)*(x1-self.x1mean)
        conditioned_variance=self.x2variance-xp.power(self.x12covariance,2.)/self.x1variance
        norm_conditioned=get_gaussian_norm(self.x2min,self.x2max,conditioned_mean,xp.sqrt(conditioned_variance))
        conditioned_log=-0.5*xp.log(2*xp.pi*conditioned_variance)-0.5*xp.power(x2-conditioned_mean,2.)/conditioned_variance-xp.log(norm_conditioned)
        y=marginal_log+conditioned_log
        y=self._check_bound_pdf(x1,x2,y)
        return y 
    
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
        x1samp=np.random.uniform(self.x1min,self.x1max,size=10000)
        x2samp=np.random.uniform(self.x2min,self.x2max,size=10000)
        pdfeval=self.pdf(x1samp,x2samp)
        idx=np.random.choice(len(x1samp),size=N,replace=True,p=pdfeval/pdfeval.sum())
        return x1samp[idx],x2samp[idx]
        
# LVK Reviewed
class PowerLawGaussian(basic_1dimpdf):
    
    def __init__(self,minpl,maxpl,alpha,lambdag,meang,sigmag,ming,maxg):
        '''
        Class for a Power Law + Gaussian probability
        
        Parameters
        ----------
        minpl,maxpl,alpha,lambdag,meang,sigmag,ming,maxg: float
            In sequence, minimum, maximum, exponential of the powerlaw part. Fraction, mean, sigma, min value, max value of the gaussian
        '''
        super().__init__(min(minpl,ming),max(maxpl,maxg))
        self.minpl,self.maxpl,self.alpha,self.lambdag,self.meang,self.sigmag,self.ming,self.maxg=minpl,maxpl,alpha,lambdag,meang,sigmag,ming,maxg
        self.PL=PowerLaw(minpl,maxpl,alpha)
        self.TG=TruncatedGaussian(meang,sigmag,ming,maxg)
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        xp = get_module_array(x)
        toret=xp.logaddexp(xp.log1p(-self.lambdag)+self.PL.log_pdf(x),xp.log(self.lambdag)+self.TG.log_pdf(x))
        return toret
    
    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: xp.array
        '''
        xp = get_module_array(x)
        toret=xp.log((1-self.lambdag)*self.PL.cdf(x)+self.lambdag*self.TG.cdf(x))
        return toret

# LVK Reviewed
class BrokenPowerLaw(basic_1dimpdf):
    
    def __init__(self,minpl,maxpl,alpha_1,alpha_2,b):
        '''
        Class for a Broken Powerlaw probability
        
        Parameters
        ----------
        minpl,maxpl,alpha_1,alpha_2,b: float
            In sequence, minimum, maximum, exponential of the first and second powerlaw part, b is at what fraction the powerlaw breaks.
        '''
        super().__init__(minpl,maxpl)
        self.minpl,self.maxpl,self.alpha_1,self.alpha_2,self.b=minpl,maxpl,alpha_1,alpha_2,b
        self.break_point = minpl+b*(maxpl-minpl)
        self.PL1=PowerLaw(minpl,self.break_point,alpha_1)
        self.PL2=PowerLaw(self.break_point,maxpl,alpha_2)
        self.norm_fact=(1+self.PL1.pdf(np.array([self.break_point]))[0]/self.PL2.pdf(np.array([self.break_point]))[0])
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        xp = get_module_array(x)
        toret=xp.logaddexp(self.PL1.log_pdf(x),self.PL2.log_pdf(x)+self.PL1.log_pdf(xp.array([self.break_point]))
            -self.PL2.log_pdf(xp.array([self.break_point])))-xp.log(self.norm_fact)
        return toret
    
    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: xp.array
        '''
        xp = get_module_array(x)
        toret=xp.log((self.PL1.cdf(x)+self.PL2.cdf(x)*
        (self.PL1.pdf(xp.array([self.break_point]))
        /self.PL2.pdf(xp.array([self.break_point]))))/self.norm_fact)
        return toret

# LVK Reviewed
class PowerLawTwoGaussians(basic_1dimpdf):
    
    def __init__(self,minpl,maxpl,alpha,lambdag,lambdaglow,meanglow,sigmaglow,minglow,maxglow,
  meanghigh,sigmaghigh,minghigh,maxghigh):
        '''
        Class for a power law + 2 Gaussians probability
        
        Parameters
        ----------
        minpl,maxpl,alpha,lambdag,lambdaglow,meanglow,sigmaglow,minglow,maxglow,
  meanghigh,sigmaghigh,minghigh,maxghigh: float
            In sequence, minimum, maximum, exponential of the powerlaw part. Fraction of pdf in gaussians and fraction in the lower gaussian.
            Mean, sigma, minvalue and maxvalue of the lower gaussian. Mean, sigma, minvalue and maxvalue of the higher gaussian 
        '''
        super().__init__(min(minpl,minglow,minghigh),max(maxpl,maxglow,maxghigh))
        self.minpl,self.maxpl,self.alpha,self.lambdag,self.lambdaglow,self.meanglow,self.sigmaglow,self.minglow,self.maxglow,\
        self.meanghigh,self.sigmaghigh,self.minghigh,self.maxghigh=minpl,maxpl,alpha,lambdag,lambdaglow,\
        meanglow,sigmaglow,minglow,maxglow,meanghigh,sigmaghigh,minghigh,maxghigh
        self.PL=PowerLaw(minpl,maxpl,alpha)
        self.TGlow=TruncatedGaussian(meanglow,sigmaglow,minglow,maxglow)
        self.TGhigh=TruncatedGaussian(meanghigh,sigmaghigh,minghigh,maxghigh)
        
    def _log_pdf(self,x):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        xp = get_module_array(x)
        pl_part = xp.log1p(-self.lambdag)+self.PL.log_pdf(x)
        g_low = self.TGlow.log_pdf(x)+xp.log(self.lambdag)+xp.log(self.lambdaglow)
        g_high = self.TGhigh.log_pdf(x)+xp.log(self.lambdag)+xp.log1p(-self.lambdaglow)
        return xp.logaddexp(xp.logaddexp(pl_part,g_low),g_high)
    
    def _log_cdf(self,x):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: xp.array
        '''
        xp = get_module_array(x)
        pl_part = (1.-self.lambdag)*self.PL.cdf(x)
        g_part =self.TGlow.cdf(x)*self.lambdag*self.lambdaglow+self.TGhigh.cdf(x)*self.lambdag*(1-self.lambdaglow)
        return xp.log(pl_part+g_part)

# LVK Reviewed
class absL_PL_inM(basic_1dimpdf):
    
    def __init__(self,Mmin,Mmax,alpha):
        super().__init__(Mmin,Mmax)
        self.Mmin=Mmin
        self.Mmax=Mmax
        self.Lmax=M2L(Mmin)
        self.Lmin=M2L(Mmax)
        
        self.L_PL=PowerLaw(self.Lmin,self.Lmax,alpha+1.)
        self.L_PL_CDF=PowerLaw(self.Lmin,self.Lmax,alpha)
        self.alpha=alpha

        self.extrafact=0.4*np.log(10)*PL_normfact(self.Lmin,self.Lmax,alpha+1.)/PL_normfact(self.Lmin,self.Lmax,alpha)
    
    def _log_pdf(self,M):
        '''
        Evaluates the log_pdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_pdf
        
        Returns
        -------
        log_pdf: xp.array
        '''
        xp = get_module_array(M)
        toret=self.L_PL.log_pdf(M2L(M))+xp.log(self.extrafact)
        return toret
    
    def _log_cdf(self,M):
        '''
        Evaluates the log_cdf
        
        Parameters
        ----------
        x: xp.array
            where to evaluate the log_cdf
        
        Returns
        -------
        log_cdf: xp.array
        '''
        xp = get_module_array(M)
        toret=xp.log(1.-self.L_PL_CDF.cdf(M2L(M)))
        return toret

class piecewise_constant_2d_distribution_normalized():

    """
    Class for a piecewise constant source frame mass distribution. 
    The 2d mass distribution is divided in a 2d checkerboard pattern, with the 
    constraint m1 > m2. The checkerboard boxes along the diagonal are cut in half. 

    Parameters
    ----------

    dist_min: float
        The minimum of the distribution
    dist_max: float
        The maximum of the distribution
    weights: 1d array
        The array determining the individual weights of each bin. 
    
    """

    def __init__(self, dist_min, dist_max, weights):

        self.dist_min, self.dist_max = dist_min, dist_max
        self.n_bins = len(weights)

        # use formula n * (n+1) / 2 to invert for n
        self.n_bins_1d = int(-1/2 + np.sqrt(1/4 + 2 * self.n_bins))

        self.weights = weights
        xp = get_module_array(weights)
    
        self.grid_x1 = xp.linspace(self.dist_min, self.dist_max, self.n_bins_1d + 1)
        self.grid_x2 = xp.linspace(self.dist_min, self.dist_max, self.n_bins_1d + 1)
        
        # compute normalization
        self.delta_bin_x1 = self.grid_x1[1] - self.grid_x1[0]
        self.delta_bin_x2 = self.grid_x2[1] - self.grid_x2[0]

        self.norm = self.compute_norm()  

        self.weights_normalized = self.norm * self.weights

    def compute_norm(self):

        """
        Computes the normalization for the PDF. 

        """

        xp = get_module_array(self.weights)

        # the weights on the diagonal should get a factor half, since they only contribute a triangle
        # Determine the indices of the upper triangle elements
        upper_triangle_indices = xp.triu_indices(self.n_bins_1d, k=0)

        # Create an empty square matrix filled with zeros
        weights_upper_triangle_matrix = xp.zeros((self.n_bins_1d, self.n_bins_1d))

        # Assign the weights to the upper triangle elements
        weights_upper_triangle_matrix[upper_triangle_indices] = self.weights
        
        for i in range(self.n_bins_1d):
            weights_upper_triangle_matrix[i,i] *= 1/2
        
        norm = 1 / xp.sum(weights_upper_triangle_matrix) * 1 / self.delta_bin_x1 / self.delta_bin_x2

        return norm

    def outside_domain_1d(self, x):

        """
        Determines elementwise if x is outside the 1d domain. 

        Parameters
        ----------
        x: array 

        Returns
        -------

        An array that is true for elements of x outside the 1d domain and false otherwise. 
        
        """

        xp = get_module_array(self.weights)

        x_smaller = x < self.dist_min 
        x_larger = x > self.dist_max
        
        return xp.logical_or(x_smaller, x_larger)

    def outside_domain_2d(self, x1, x2):

        """
        Determines whether the elemnts in the arrays x1 and x2 are outside the 1d domain and
        whether x1 < x2 (elementwise). The two arrays must be of the same shape. 

        Parameters
        ----------
        x1: array 
        x2: array 

        Returns
        -------

        An array that is true if either the respective x1 or x2 elements are outside 
        the 1d domain and if x1 < x2 (elementwise). 
        
        """

        # check whether x1 (or x2) is elementwise outside the domain
        x1_outside = self.outside_domain_1d(x1)
        x2_outside = self.outside_domain_1d(x2)

        # check whether x1 is smaller than x2 (elementwise)
        x1_smaller_than_x2 = x1 < x2

        # determine the elemtwise pairs of x1 and x2 that are outside the domain 
        point_outside_grid = xp.logical_or(x1_outside, x2_outside)
        
        # additionally return only true if x1 < x2 (elementwise)
        return xp.logical_or(point_outside_grid, x1_smaller_than_x2)

    def compute_conditions(self, positions):

        return [positions == i for i in range(self.n_bins)]

    def compute_flat_position(self, position_in_grid):

        """
        Computes the numbered bin for a 2d tuple (see example below for the numbering convention). 

        We want to go from the 2d positions in the triangle to a unique numbering in 1d.

        For example: 

        No    | No    | (2,2)
        No    | (1,1) | (2,1) 
        (0,0) | (1,0) | (2,0) 

        To the numbering
        No| No| 5
        No| 3 | 4
        0 | 1 | 2

        Parameters
        ----------
        position_in_grid 2d tuple
            The tuple determining the position in the grid. 
        
        """

        flat_position_in_square = position_in_grid[0] + self.n_bins_1d * position_in_grid[1]
        subtract_triangle_constraint = position_in_grid[1] * (position_in_grid[1] + 1) / 2
        
        return flat_position_in_square - subtract_triangle_constraint
    
    def pdf(self, x1, x2):

        """
        Computes the PDF for the arrays x1, x2

        Parameters
        ----------
        x1: array 
        x2: array 

        Returns
        -------
        An array of the PDF evaluated at p(x1, x2)
        
        """

        xp = get_module_array( self.weights_normalized)

        position_in_grid = self.determine_grid_position(x1, x2)

        # compute the arbitrary number that corresponds to the unique weight
        positions_flat = self.compute_flat_position(position_in_grid)

        self.conditions = self.compute_conditions(positions_flat)

        self.pdf_func = lambda x1: xp.piecewise(x1, self.conditions, self.weights_normalized)

        pdf = self.pdf_func(positions_flat)
        
        return xp.where(self.outside_domain_2d(x1, x2), 0, pdf)

    def log_pdf(self, x1, x2):

        """
        Computes the logarithm of the PDF for the arrays x1, x2

        Parameters
        ----------
        x1: array 
        x2: array 

        Returns
        -------
        An array of the log PDF evaluated at p(x1, x2)
        
        """
         
        xp = get_module_array(x1)
    
        return xp.log(self.pdf(x1, x2))

    def determine_grid_position(self, x1, x2):

        n1 = (x1 - self.dist_min) // self.delta_bin_x1
        n2 = (x2 - self.dist_min) // self.delta_bin_x2
        
        return (n1, n2)

