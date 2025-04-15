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
from training import train, dataset_maker
import forcing


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



# Location
point_loc = [-47.4,34.6]
R = 5. # 5.0 

# my forward model
dt = 60.

# NN hyper parameters
LEARNING_RATE = 1e-2
SEED = 5678
MAX_STEP = 20
PRINT_EVERY = 10




key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)



filename = '../../data_regrid/croco_1h_inst_surf_2005-05-01-2005-05-31_0.1deg_conservative.nc'
ds = xr.open_mfdataset(filename)
#ds = ds.sel(lon=point_loc[0],lat=point_loc[1], method='nearest')
Nsize = 32 # 256
ds = ds.isel(lon=slice(-Nsize,-1),lat=slice(-Nsize,-1))


LON_bounds = [point_loc[0]-R,point_loc[0]+R]
LAT_bounds = [point_loc[1]-R,point_loc[1]+R]


# forcing
TAx = jnp.asarray(ds.oceTAUX.values, dtype='float')
TAy = jnp.asarray(ds.oceTAUY.values, dtype='float')
Ug = jnp.asarray(ds.Ug.values, dtype='float')
Vg = jnp.asarray(ds.Vg.values, dtype='float')
fc = jnp.asarray(2*2*np.pi/86164*np.sin(ds.lat.values*np.pi/180))
dt_forcing = 3600.

# dataloader
Nhours = 10*24 # 45*24 
train_data, test_data = dataset_maker(ds, Nhours, dt_forcing)



# control
K0 = jnp.asarray(-10., dtype='float')
dissipation_model = DissipationNN(subkey)

# NN initialisation, required in this hybrid model
# "ReZero"
# dissipation_model = eqx.tree_at( lambda t:t.layer1.weight, dissipation_model, dissipation_model.layer1.weight/1e5)
#dissipation_model = eqx.tree_at( lambda t:t.layer1.bias, dissipation_model, dissipation_model.layer1.bias*0.)
dissipation_model = eqx.tree_at( lambda t:t.layer2.weight, dissipation_model, dissipation_model.layer2.weight*0.) # <- idea from Farchi et al. 2021
dissipation_model = eqx.tree_at( lambda t:t.layer2.bias, dissipation_model, dissipation_model.layer2.bias*0.)
# runtime
# t0 = 0.0
# t1 = len(ds.time)*dt_forcing
# 
# call_args = [t0, t1]


# model definition
mymodel = jslab( K0, TAx, TAy, Ug, Vg, fc, dt_forcing, dissipation_model, dt)
# optimiser
optim = optax.adam(LEARNING_RATE)
# training
print('* Training...')
model = train(mymodel, optim, train_data, test_data, maxstep=MAX_STEP, print_every=PRINT_EVERY)
print('     done !')





print('* Plotting')

U_verif,_ = model(test_data['t0'],test_data['t1'])
U_true = test_data['U']

fig, ax = plt.subplots(1,2,figsize = (10,5),constrained_layout=True,dpi=100)
ax[0].pcolormesh(U_verif[-1], vmin=-0.6,vmax=0.6)
ax[0].set_title('U on test_data')
ax[1].pcolormesh(U_true[-1], vmin=-0.6,vmax=0.6)
ax[1].set_title('True U')

# t0_t, t1_t = test_data['t0'], test_data['t1']
# time = np.arange(t0_t, t1_t, dt_forcing)/(86400)
# U_before,_ = mymodel(t0_t,t1_t)
# U_after, _ = model(t0_t,t1_t)
# fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
# ax.plot( time, test_data['U'][:,5,5], label='target', c='k')
# ax.plot( time, U_before[:,5,5], label='before', ls='--') 
# ax.plot( time, U_after[:,5,5], label='after') 
# ax.set_xlabel('days')
# ax.set_ylabel('U')
# ax.grid()
# ax.set_title('test data')
# plt.legend()

# t0_tr, t1_tr = train_data['t0'], train_data['t1']
# time2 = np.arange(t0_tr, t1_tr, dt_forcing)/(86400)
# U_before,_ = mymodel(t0_tr,t1_tr)
# U_after, _ = model(t0_tr,t1_tr)
# fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
# ax.plot( time2, train_data['U'][:,5,5], label='target', c='k')
# ax.plot( time2, U_before[:,5,5], label='before', ls='--') 
# ax.plot( time2, U_after[:,5,5], label='after') 
# ax.set_xlabel('days')
# ax.set_ylabel('U')
# ax.grid()
# ax.set_title('train data')
# plt.legend()




plt.show()