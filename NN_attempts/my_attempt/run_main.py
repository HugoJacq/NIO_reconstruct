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

LEARNING_RATE = 1e-2
SEED = 5678

key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
dissipation_model = DissipationNN(subkey)


point_loc = [-47.4,34.6]
ds = xr.open_mfdataset('../../data_regrid/croco_1h_inst_surf_2005-05-01-2005-05-31_0.1deg_conservative.nc')
#ds = ds.sel(lon=point_loc[0],lat=point_loc[1], method='nearest')
Nsize = 128 # 256
ds = ds.isel(lon=slice(200,200+Nsize),lat=slice(72,72+Nsize))


# forcing
TAx = jnp.asarray(ds.oceTAUX.values, dtype='float')
TAy = jnp.asarray(ds.oceTAUY.values, dtype='float')
fc = jnp.asarray(2*2*np.pi/86164*np.sin(ds.lat.values*np.pi/180))
dt_forcing = 3600.


# dataloader
Nhours = 10*24 # 45*24 
train_data, test_data = dataset_maker(ds, Nhours, dt_forcing)



# control
K0 = jnp.asarray(-10., dtype='float')
dissipation_model = DissipationNN(subkey)
dissipation_model = eqx.tree_at( lambda t:t.layer1.weight, dissipation_model, dissipation_model.layer1.weight*0.0)
dissipation_model = eqx.tree_at( lambda t:t.layer1.bias, dissipation_model, dissipation_model.layer1.bias*0.0)
# dissipation_model = eqx.tree_at( lambda t:t.layer2.weight, dissipation_model, dissipation_model.layer2.weight*0.0)
# dissipation_model = eqx.tree_at( lambda t:t.layer3.weight, dissipation_model, dissipation_model.layer3.weight*0.0)
# runtime
t0 = 0.0
t1 = len(ds.time)*dt_forcing
dt = 60.
call_args = [t0, t1]


# model definition
# mymodel = jslab( K0, TAx, TAy, fc, dt_forcing, dissipation_model, call_args)
mymodel = jslab( K0, TAx, TAy, fc, dt_forcing, dissipation_model, dt)

# U_before,V_before = mymodel()
U_test, t0_t, t1_t = test_data['U'], test_data['t0'], test_data['t1']
print(t0_t/3600, t1_t/3600)
# U_before,V_before = mymodel(call_args=[t0_t,t1_t])
U_before,V_before = mymodel(t0_t,t1_t)

fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
plt.show()

# optimiser
optim = optax.adam(LEARNING_RATE)



# training
print('* Training...')
model = train(mymodel, optim, train_data, test_data, maxstep=50, print_every=10)
print('     done !')


# U_after, V_after = model(call_args=[t0_t,t1_t])
U_after, V_after = model(t0_t,t1_t)

time = np.arange(t0_t, t1_t, dt_forcing)/(86400)

print('* Plotting')
fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=100)
ax.plot( time, test_data['U'][:,5,5], label='target', c='k')
ax.plot( time, U_before[:,5,5], label='before', ls='--') 
ax.plot( time, U_after[:,5,5], label='after') 
ax.set_xlabel('days')
ax.set_ylabel('U')
ax.grid()
plt.legend()

plt.show()