import numpy as np
import scipy as sn

CUPY_LOADED = False

def enable_cupy():
    try:
        import cupy as cp
        import cupyx as _cupyx
        from cupyx.scipy import interpolate
        from cupy import _core

        global cp
        global _cupyx
        global interpolate
        global _core
        global CUPY_LOADED
        
        CUPY_LOADED = True
        print('cupy imported')
    except ImportError:
        print('Error in importing cupy')

def disable_cupy():
    global CUPY_LOADED
    CUPY_LOADED = False
    
def is_there_cupy():
    return CUPY_LOADED

try:
    import config
    print('Config file loaded')
    if config.CUPY:
        enable_cupy()
    else:
        print('Config does not allow CUPY')        
except ImportError:
    print('Config not imported, automatically decides between Numpy and Cupy')
    enable_cupy()

def cp2np(array):
    '''Cast any array to numpy'''
    if CUPY_LOADED:
        return cp.asnumpy(array)
    else:
        return array
        

def np2cp(array):
    '''Cast any array to cupy'''
    if CUPY_LOADED:
        return cp.asarray(array)
    else:
        return array

def get_module_array(array):
    if CUPY_LOADED:
        return cp.get_array_module(array)
    else:
        return np

def get_module_array_scipy(array):
    if CUPY_LOADED:
        return _cupyx.scipy.get_array_module(array)
    else:
        return sn

def iscupy(array):
    if CUPY_LOADED:
        return isinstance(array, (cp.ndarray, _cupyx.scipy.sparse.spmatrix,
                        _core.fusion._FusionVarArray,
                        _core.new_fusion._ArrayProxy))
    else:
        return False


