"""
Goal of this script:

Use a NN to predict the total current of next time step.
This is a example to show:
    - how to used JAX and Equinox to build a NN
    - how to train a NN
    - demonstrate that a CNN (convolutional NN) can reconstructe the total current
        given maps of SSH and wind stress
        
But this example is also:
    - a proof of concept, results should not be taken as publication ready
    - a toy example to play with different NN architectures
    - a preliminary example for a more complex NN: modeling dissipation inside a equation


INPUT of the NN: Ug, Vg, Taux, Tauy
OUPUT of the NN: U total,V total

I give the inputs at time t, and i want to predict U,V at time t+dt.
My data is 1 month, so i give 'Ntime' input (from t0 to t0+Ntime*dt) 
    and i train on the ouputs at t0+dt to Ntime*(dt+1).
    So that my ouput are shifted in time by 1 dt.

My test data is using the same structure on the rest of the data in the month

Note: when running the script, JAX/XLA print many warnings in the terminal. Is it important ?

Author: Hugo Jacquet, April 2025
"""

import jax
import xarray as xr
import numpy as np
import optax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import equinox as eqx 
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" # for jax

from model import blackbox
from train import do_training, evaluate, loss

# hyper parameters
Nsize = 128             # size of the nx,ny
Ntime = 512             # numbe of timestep to train on
LEARNING_RATE = 1e-2    # for the optimizer
SEED = 5678             # for reproducibility
MAX_STEP = 200          # for the train loop
PRINT_EVERY = 10        # for the train loop
TRAIN = False

# input data
filename = '../../data_regrid/croco_1h_inst_surf_2005-05-01-2005-05-31_0.1deg_conservative.nc'
ds = xr.open_mfdataset(filename)
ds = ds.isel(lon=slice(-Nsize-1,-1),lat=slice(-Nsize-1,-1)) # reduce the dataset to lower right corner
nt = len(ds.time)

# INFOS
print(f'total time instant : {nt}')
print(f'instant used for training : {Ntime}')
print(f'instants used for validation : {nt-Ntime}')


# forcings
TAUx = ds.oceTAUX.values
TAUy = ds.oceTAUY.values
Ug = ds.Ug.values
Vg = ds.Vg.values

# true fields
U = ds.U.values
V = ds.V.values
Ut = U+Ug
Vt = V+Vg

# Making datasets
obs = {}
features = {}

features['train'] = np.stack([Ug[:Ntime], Vg[:Ntime], TAUx[:Ntime], TAUy[:Ntime]])  # shape : features, time, ny, nx
obs['train'] = np.stack([Ut[1:Ntime+1],Vt[1:Ntime+1]])                              # shape : current, time, ny, nx
features["val"] = np.stack([Ug[-(nt-Ntime):-1], Vg[-(nt-Ntime):-1], TAUx[-(nt-Ntime):-1], TAUy[-(nt-Ntime):-1]])
obs['val'] = np.stack([Ut[-(nt-Ntime)+1:],Vt[-(nt-Ntime)+1:]])

# initialize the model
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
model = blackbox(key, Ninput_features=features['train'].shape[0])

# run the training loop
if TRAIN:
    optim = optax.adam(LEARNING_RATE)
    last_model, best_model, Ltrain_loss, Ltest_loss = do_training(model, optim, features, obs, MAX_STEP, PRINT_EVERY )
    # savin weight and bias as binary file
    eqx.tree_serialise_leaves('./myNN.pt',best_model)
    
    fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
    epochs = np.arange(len(Ltrain_loss))
    ax.plot(epochs, Ltrain_loss, label='train')
    ax.plot(epochs, Ltest_loss, label='test')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig('Loss.png')


trained_model = eqx.tree_deserialise_leaves('./myNN.pt', blackbox(key, Ninput_features=features['train'].shape[0]))

# inference
C_pred = jax.vmap(trained_model, in_axes=1, out_axes=1)(features['train'])
C_val = jax.vmap(trained_model, in_axes=1, out_axes=1)(features['val'])
test_loss = evaluate(model, features['val'], obs['val'])
train_loss = loss(model, features['train'], obs['train'])
print(f'test_loss={test_loss}')
print(f'train_loss={train_loss}')
# plots


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