import jax
from jax import config as jconf
jconf.update("jax_enable_x64", True)
print('ciao')

try:
    import config
    print('Config file loaded')
    if not config.CUPY: 
        jax.config.update('jax_platform_name', 'cpu')
except:
    print('No configuration file')

import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as onp
import scipy as oscipy

from jax.scipy.special import erf, betaln, betainc, gamma, logsumexp, spence
from jax.scipy.stats import gaussian_kde

def beta(a,b):
    return jnp.exp(betaln(a,b))

def jnp2onp(array):
    return onp.asarray(array)

def onp2jnp(array):
    return jnp.asarray(array)


from typing import Iterable, Optional

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

def interp2d(
    x: jnp.ndarray,
    y: jnp.ndarray,
    xp: jnp.ndarray,
    yp: jnp.ndarray,
    zp: jnp.ndarray,
    fill_value: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Bilinear interpolation on a grid. ``CartesianGrid`` is much faster if the data
    lies on a regular grid.

    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`

    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.
    """
    if xp.ndim != 1 or yp.ndim != 1:
        raise ValueError("xp and yp must be 1D arrays")
    if zp.shape != (xp.shape + yp.shape):
        raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")

    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

    # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    if fill_value is not None:
        oob = jnp.logical_or(
            x < xp[0], jnp.logical_or(x > xp[-1], jnp.logical_or(y < yp[0], y > yp[-1]))
        )
        z = jnp.where(oob, fill_value, z)

    return z