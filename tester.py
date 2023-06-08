from icarogw.jax_pal import *
from astropy.cosmology import Planck15
from icarogw.cosmology import astropycosmology
import healpy as hp


print(hp.pix2ang(32,jnp.arange(0,100,1).astype(int)))
