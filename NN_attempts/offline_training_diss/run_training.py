import jax
import xarray as xr
import numpy as np
import optax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import equinox as eqx 
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax

from model import RHS, DissipationNN



"""
I do 'resolvant correction' from Farchi et al. 2021 https://doi.org/10.1016/j.jocs.2021.101468 
            "A comparison of combined data assimilation and machine learning methods for offline and online model error correction"
"""


# NN hyper parameters
LEARNING_RATE = 1e-2
SEED = 5678
MAX_STEP = 20
PRINT_EVERY = 10

filename = '../../data_regrid/croco_1h_inst_surf_2005-05-01-2005-05-31_0.1deg_conservative.nc'
ds = xr.open_mfdataset(filename)
Nsize = 32 # 256
ds = ds.isel(lon=slice(-Nsize,-1),lat=slice(-Nsize,-1))


true_dCdt = jnp.gradient(ds.U.values), jnp.gradient(ds.V.values)

U = jnp.asarray(ds.U.values, dtype='float')
V = jnp.asarray(ds.V.values, dtype='float')
TAx = jnp.asarray(ds.oceTAUX.values, dtype='float')
TAy = jnp.asarray(ds.oceTAUY.values, dtype='float')
fc = jnp.asarray(2*2*np.pi/86164*np.sin(ds.lat.values*np.pi/180))
Tau = TAx,TAy
C = U,V

K = jnp.asarray(-10., dtype='float')


key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
mydissNN = DissipationNN(subkey)


myRHS = RHS(C, fc, Tau, K, mydissNN)