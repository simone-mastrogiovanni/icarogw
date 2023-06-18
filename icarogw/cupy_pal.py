try:
    import config
    print('Config file loaded')
    if config.CUPY:
        try:
            import cupy as cp
            import numpy as np
            import cupyx as _cupyx
            from cupyx.scipy import interpolate
            from cupy import _core
            import scipy as sn
            
            CUPY_LOADED = True
            print('CUPY LOADED')
        except ImportError:
            import numpy as np
            import scipy as sn
            CUPY_LOADED = False
            print('CUPY NOT LOADED BACK TO NUMPY')
    else:
        import numpy as np
        import scipy as sn
        CUPY_LOADED = False
        print('CUPY NOT LOADED')        
except ImportError:
    print('Config not imported, automatically decides between Numpy and Cupy')
    try:
        import numpy as np
        import cupy as cp
        import cupyx as _cupyx
        from cupyx.scipy import interpolate
        from cupy import _core
        import scipy as sn
        CUPY_LOADED = True
        print('CUPY LOADED')
    except ImportError:
        import numpy as np
        import scipy as sn
        CUPY_LOADED = False
        print('CUPY NOT LOADED BACK TO NUMPY')

if CUPY_LOADED: 
    def cp2np(array):
        '''Cast any array to numpy'''
        return cp.asnumpy(array)

    def np2cp(array):
        '''Cast any array to cupy'''
        return cp.asarray(array)
    
    def get_module_array(array):
        return cp.get_array_module(array)
    
    def get_module_array_scipy(array):
        
        return _cupyx.scipy.get_array_module(array)
    
    def iscupy(array):
        return isinstance(array, (cp.ndarray, _cupyx.scipy.sparse.spmatrix,
                            _core.fusion._FusionVarArray,
                            _core.new_fusion._ArrayProxy))
else:
    def cp2np(array):
        '''Cast any array to numpy'''
        return array
    
    def np2cp(array):
        '''Cast any array to cupy'''
        return array
    
    def get_module_array(array):
        return np
    
    def get_module_array_scipy(array):
        return sn
    
    def iscupy(array):
        return False
