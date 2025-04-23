import jax
import xarray as xr
import numpy as np
import optax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import equinox as eqx 
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax

from model import DissipationNN, jslab
from training import train, dataset_maker, evaluate
import forcing
from constants import oneday

"""
Things i need to do:

- normalize inputs of NN, then de-normalize its ouput
- use a dataloader with batches (faster training)
- save parameters of the best model during training (at min of loss(train_data) )
- during training, start with a few timestep then increase the time horizon. -> interactive code ?
- tweak NN architecture: use CNN or MLP.


I do 'tendency correction' from Farchi et al. 2021 https://doi.org/10.1016/j.jocs.2021.101468 
            "A comparison of combined data assimilation and machine learning methods for offline and online model error correction"
"""

TRAIN = False

# Location
point_loc = [-47.4,34.6]    # °W °N
R = 5.                      # in degrees

# my forward model
dt = 60.                    # seconds

# NN hyper parameters
LEARNING_RATE = 1e-2
SEED = 5678
MAX_STEP = 10
PRINT_EVERY = 1




key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)



filename = ['../../data_regrid/croco_1h_inst_surf_2005-05-01-2005-05-31_0.1deg_conservative.nc',]
ds = xr.open_mfdataset(filename)
nt = len(ds.time)
#ds = ds.sel(lon=point_loc[0],lat=point_loc[1], method='nearest')
Nsize = 32 # 256
Nhours = 10*24 # number of dt used for test_data
ds = ds.isel(lon=slice(-Nsize,-1),lat=slice(-Nsize,-1))

# INFOS
print('*** INFOS ****')
print(f'total time instant : {nt}')
print(f'instant used for training : {nt - Nhours}')
print(f'instants used for validation : {Nhours}')
print(f'tile size: {Nsize} points squared')
print('*******')
print('')

LON_bounds = [point_loc[0]-R,point_loc[0]+R]
LAT_bounds = [point_loc[1]-R,point_loc[1]+R]


# forcing
TAx = jnp.asarray(ds.oceTAUX.values, dtype='float')
TAy = jnp.asarray(ds.oceTAUY.values, dtype='float')
Ug = jnp.asarray(ds.Ug.values, dtype='float')
Vg = jnp.asarray(ds.Vg.values, dtype='float')
U = jnp.asarray(ds.U.values, dtype='float')
V = jnp.asarray(ds.V.values, dtype='float')
fc = jnp.asarray(2*2*np.pi/86164*np.sin(ds.lat.values*np.pi/180))
dt_forcing = 3600.

# dataloader
train_data, test_data = dataset_maker(ds, Nhours, dt_forcing)

# control
K0 = jnp.asarray(-10., dtype='float')
features = jnp.stack([Ug, Vg, U, V], axis=1)
dissipation_model = DissipationNN(subkey, Nfeatures=features.shape[1])

# NN initialisation, required in this hybrid model
# "ReZero"
# dissipation_model = eqx.tree_at( lambda t:t.layer1.weight, dissipation_model, dissipation_model.layer1.weight/1e5)
#dissipation_model = eqx.tree_at( lambda t:t.layer1.bias, dissipation_model, dissipation_model.layer1.bias*0.)
# dissipation_model = eqx.tree_at( lambda t:t.layer2.weight, dissipation_model, dissipation_model.layer2.weight*0.) # <- idea from Farchi et al. 2021
# dissipation_model = eqx.tree_at( lambda t:t.layer2.bias, dissipation_model, dissipation_model.layer2.bias*0.)
# runtime
# t0 = 0.0
# t1 = len(ds.time)*dt_forcing
# 
# call_args = [t0, t1]


# model definition

mymodel = jslab( K0, TAx, TAy, features, fc, dt_forcing, dissipation_model, dt)

if TRAIN:
    # optimiser
    optim = optax.adam(LEARNING_RATE)
    # training
    print('* Training...')
    _, best_model, Train_loss, Test_loss = train(mymodel, optim, train_data, test_data, maxstep=MAX_STEP, print_every=PRINT_EVERY)
    print('     done !')
    eqx.tree_serialise_leaves('./mybest_model.pt',best_model)
    
    fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
    epochs = np.arange(len(Train_loss))
    ax.plot(epochs, Train_loss, label='train')
    ax.plot(epochs, Test_loss, label='test')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig('Loss.png')

best_model = eqx.tree_deserialise_leaves('./mybest_model.pt', jslab( K0, TAx, TAy, features, fc, dt_forcing, dissipation_model, dt))
train_loss = evaluate(best_model, test_data)
test_loss = evaluate(best_model, train_data)
print('best_model:')
print(f'train_loss = {train_loss}')
print(f'test_loss = {test_loss}')



# infos about the NN dissipation
print('layer2.weight',best_model.dissipation_model.layer2.weight)
print('layer2.bias',best_model.dissipation_model.layer2.bias)


print('* Plotting')

# mean trajectory of zonal current on test_data
mean_U_test = best_model(test_data['t0'],test_data['t1'])[0].mean(axis=(1,2))
mean_U_true = ds.U.isel(time=slice(- Nhours-1,-1)).mean(axis=(1,2))
time_test = np.arange(test_data['t0'], test_data['t1'], int(dt_forcing))/oneday
fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
ax.plot(time_test, mean_U_test, label='estimated <U>xy')
ax.plot(time_test, mean_U_true, label='true <U>xy')
ax.set_title('test data')
ax.set_xlabel('time (days)')
ax.set_ylabel('U (m/s)')
ax.legend()
fig.savefig('meanXY_Utraj_test_data.png')

# mean trajectory of zonal current on train_data
mean_U_train = best_model(train_data['t0'],train_data['t1'])[0].mean(axis=(1,2))
mean_U_true = ds.U.isel(time=slice(0,- Nhours)).mean(axis=(1,2))
time_train = np.arange(train_data['t0'], train_data['t1'], int(dt_forcing))/oneday
fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
ax.plot(time_train, mean_U_train, label='estimated <U>xy')
ax.plot(time_train, mean_U_true, label='true <U>xy')
ax.set_title('train data')
ax.set_xlabel('time (days)')
ax.set_ylabel('U (m/s)')
ax.legend()
fig.savefig('meanXY_Utraj_train_data.png')

# last snapshot of tile
U_verif,_ = best_model(test_data['t0'],test_data['t1'])
U_true = test_data['U']
fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=100)
ax[0].pcolormesh(U_verif[-1], vmin=-0.6,vmax=0.6)
ax[0].set_title('U on test_data')
ax[1].pcolormesh(U_true[-1], vmin=-0.6,vmax=0.6)
ax[1].set_title('True U')
fig.savefig('last_time_tile.png')






plt.show()