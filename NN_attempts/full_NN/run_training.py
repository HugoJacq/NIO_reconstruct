import jax
import xarray as xr
import numpy as np
import optax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import equinox as eqx 
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax


from model import blackbox
from train import do_training

"""
Goal of this script:

Try to use a NN to predict the total current of next time step.
working only on 1 time step for now


"""




Nsize = 128 # 256
Ntime = 512 # 60
LEARNING_RATE = 1e-2
SEED = 5678
MAX_STEP = 500
PRINT_EVERY = 1


filename = '../../data_regrid/croco_1h_inst_surf_2005-05-01-2005-05-31_0.1deg_conservative.nc'
ds = xr.open_mfdataset(filename)
# ds = ds.isel(lon=slice(-Nsize-1,-1),lat=slice(-Nsize-1,-1), time=slice(0,Ntime))
ds = ds.isel(lon=slice(-Nsize-1,-1),lat=slice(-Nsize-1,-1))
nt = len(ds.time)
print(nt)

print(f'total time instant : {nt}')
print(f'instant used for training : {Ntime}')
print(f'instants used for validation : {nt-Ntime}')

# true fields
U = ds.U.values
V = ds.V.values
Ug = ds.Ug.values
Vg = ds.Vg.values

Ut = U+Ug
Vt = V+Vg

# forcings
TAUx = ds.oceTAUX.values
TAUy = ds.oceTAUY.values

obs = {}
features = {}


features['train'] = np.stack([Ug[:Ntime], Vg[:Ntime], TAUx[:Ntime], TAUy[:Ntime]])
obs['train'] = np.stack([Ut[1:Ntime+1],Vt[1:Ntime+1]]) # current, time, ny, nx

features["val"] = np.stack([Ug[-(nt-Ntime):-1], Vg[-(nt-Ntime):-1], TAUx[-(nt-Ntime):-1], TAUy[-(nt-Ntime):-1]])
obs['val'] = np.stack([Ut[-(nt-Ntime)+1:],Vt[-(nt-Ntime)+1:]])

key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
model = blackbox(key, Ninput_features=features['train'].shape[0])

optim = optax.adam(LEARNING_RATE)
last_model, best_model, Ltrain_loss, Ltest_loss = do_training(model, optim, features, obs, MAX_STEP, PRINT_EVERY )
print('saving weight and bias...')
eqx.tree_serialise_leaves('./myNN.pt',best_model)

trained_model = eqx.tree_deserialise_leaves('./myNN.pt', blackbox(key, Ninput_features=features['train'].shape[0]))

C_pred = jax.vmap(trained_model, in_axes=1, out_axes=1)(features['train'])
C_val = jax.vmap(trained_model, in_axes=1, out_axes=1)(features['val'])


fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
epochs = np.arange(len(Ltrain_loss))
ax.plot(epochs, Ltrain_loss, label='train')
ax.plot(epochs, Ltest_loss, label='test')
ax.set_xlabel('epochs')
ax.set_ylabel('Loss')
ax.legend()


fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=100)
ax[0].pcolormesh(C_pred[0,-1],vmin=-0.6,vmax=0.6)
ax[0].set_title('U estimate from train features')
ax[1].pcolormesh(obs['train'][0,-1],vmin=-0.6,vmax=0.6)
ax[1].set_title('U truth, training')

fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=100)
ax[0].pcolormesh(C_val[0,-1],vmin=-0.6,vmax=0.6)
ax[0].set_title('U estimate from val features')
ax[1].pcolormesh(obs['val'][0,-1],vmin=-0.6,vmax=0.6)
ax[1].set_title('U truth, validation')
plt.show()