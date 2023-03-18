try:
    import config
    print('Config file loaded')
    if config.CUPY:
        try:
            import cupy as xp
            import numpy as np
            from cupy import trapz
            from cupyx.scipy.special import erf, beta, betainc, gamma# noqa
            from cupyx.scipy.interpolate import interpn
            CUPY_LOADED = True
            print('CUPY LOADED')
        except ImportError:
            import numpy as xp
            import numpy as np
            from numpy import trapz
            from scipy.special import erf, beta, betainc, gamma # noqa
            from scipy.interpolate import interpn
            CUPY_LOADED = False
            print('CUPY NOT LOADED BACK TO NUMPY')
    else:
        import numpy as xp
        import numpy as np
        from numpy import trapz
        from scipy.special import erf, beta, betainc, gamma # noqa
        from scipy.interpolate import interpn
        CUPY_LOADED = False
        print('CUPY NOT LOADED')        
except ImportError:
    print('Config not imported, automatically decides between Numpy and Cupy')
    try:
        import cupy as xp
        import numpy as np
        from cupy import trapz
        from cupyx.scipy.special import erf, beta, betainc, gamma  # noqa
        from cupyx.scipy.interpolate import interpn
        CUPY_LOADED = True
        print('CUPY LOADED')
    except ImportError:
        import numpy as xp
        import numpy as np
        from numpy import trapz
        from scipy.special import erf, beta, betainc, gamma # noqa
        from scipy.interpolate import interpn
        CUPY_LOADED = False
        print('CUPY NOT LOADED BACK TO NUMPY')

        
if CUPY_LOADED: 
    def cp2np(array):
        '''Cast any array to numpy'''
        return xp.asnumpy(array)

    def np2cp(array):
        '''Cast any array to cupy'''
        return xp.asarray(array)
else:
    def cp2np(array):
        '''Cast any array to numpy'''
        return array
    
    def np2cp(array):
        '''Cast any array to cupy'''
        return array

    
import itertools
    



def find_histoplace(arr,edges, clean_outliers=False):
    '''

    Parameters
    ----------
    arr: xp.array
        1-D array of values to place in the histogram
    edges: xp.array
        Monothonic increasing array of edges
    clean_outliers: bool
        If True remove the samples falling outside the edges

    Returns
    -------
    xp.array of indeces, indicating where to place them in the histogram.
    It has -1 if value is lower than lower boundary, len(edges) if is above
    '''

    # Values equal or above the first edge will be at 1, values below the first edge at 0
    # Values equal or above the last edge will be at len(edges)
    indices=xp.digitize(arr,edges,right=False)

    # Values equal or above the first edge will be at 0, values below the first edge at -1
    # Values in other bins are correctly placed
    indices[arr<edges[-1]]=indices[arr<edges[-1]]-1

    # Values that correspond to the last edge will be places at len(edges)-2
    # The motivation is that bins indeces go from 0 to len(edges-2)
    indices[arr==edges[-1]]=indices[arr==edges[-1]]-2

    if clean_outliers:
        indices=indices[(indices!=-1) & (indices!=len(edges))]

    return indices


def betaln(alpha, beta):
    '''
    Logarithm of the Beta function
    .. math::
        \\ln B(\\alpha, \\beta) = \\frac{\\ln\\gamma(\\alpha)\\ln\\gamma(\\beta)}{\\ln\\gamma(\\alpha + \\beta)}

    Parameters
    ----------
    alpha: float
        The Beta alpha parameter (:math:`\\alpha`)
    beta: float
        The Beta beta parameter (:math:`\\beta`)
    
    Returns
    -------
    ln_beta: float, array-like
        The ln Beta function
    '''
    ln_beta = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    return ln_beta

