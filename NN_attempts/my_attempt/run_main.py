import jax
import xarray as xr
import numpy as np
import optax
import matplotlib.pyplot as plt
import jax.numpy as jnp

from model import DissipationNN, jslab
from training import train, dataset_maker

LEARNING_RATE = 3e-4

SEED = 5678
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
dissipation_model = DissipationNN(subkey)


point_loc = [-47.4,34.6]
ds = xr.open_mfdataset('../../data_regrid/croco_1h_inst_surf_2005-05-01-2005-05-31_0.1deg_conservative.nc')
#ds = ds.sel(lon=point_loc[0],lat=point_loc[1], method='nearest')
Nsize = 16 # 256
ds = ds.isel(lon=slice(200,200+Nsize),lat=slice(72,72+Nsize))

# dataloader
Nhours = 10*24 # 45*24 
train_data, test_data, _ = dataset_maker(ds, Nhours)

# forcing
TAx = jnp.asarray(ds.oceTAUX.values, dtype='float')
TAy = jnp.asarray(ds.oceTAUY.values, dtype='float')
fc = jnp.asarray(2*2*np.pi/86164*np.sin(ds.lat.values*np.pi/180))
dt_forcing = 3600.

# control
K0 = jnp.asarray(-10., dtype='float')
dissipation_model = DissipationNN(subkey)

# runtime
t0 = 0.0
t1 = len(ds.time)*dt_forcing
dt = 60.
call_args = t0, t1, dt


# model definition
mymodel = jslab( K0, TAx, TAy, fc, dt_forcing, dissipation_model, call_args)


U,V = mymodel()

print(U.shape)
plt.plot(np.arange(U.shape[0]), U[:,5,5]) 
plt.show()
# optimiser
optim = optax.adam(LEARNING_RATE)


model = train(mymodel, optim, train_data, test_data, maxstep=1000, print_every=10)