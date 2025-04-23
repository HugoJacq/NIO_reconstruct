import jax
import xarray as xr
import numpy as np
import optax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import equinox as eqx 
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
jax.config.update("jax_enable_x64", True)
from model import RHS, DissipationNN, Forward_Euler
from training import dataloader, train, evaluate


"""
I do 'resolvant correction' from Farchi et al. 2021 https://doi.org/10.1016/j.jocs.2021.101468 
            "A comparison of combined data assimilation and machine learning methods for offline and online model error correction"
"""

TRAIN = False

# NN hyper parameters
LEARNING_RATE = 1e-2
SEED = 5678
MAX_STEP = 1_000
PRINT_EVERY = 10
BATCH_SIZE = 100

Ntime_test = 10*24       # number of hours for test (over the data)
Nsize = 128         # size of the domain, in nx ny
dt_forcing = 3600   # time step of forcing
dt = 3600.          # time step for Euler integration

filename = ['../../data_regrid/croco_1h_inst_surf_2005-01-01-2005-01-31_0.1deg_conservative.nc',
            '../../data_regrid/croco_1h_inst_surf_2005-02-01-2005-02-28_0.1deg_conservative.nc',
            '../../data_regrid/croco_1h_inst_surf_2005-03-01-2005-03-31_0.1deg_conservative.nc',
            "../../data_regrid/croco_1h_inst_surf_2005-04-01-2005-04-30_0.1deg_conservative.nc"]
ds = xr.open_mfdataset(filename)

ds = ds.isel(lon=slice(-Nsize-1,-1),lat=slice(-Nsize-1,-1))
nt = len(ds.time)
Ntime = nt - Ntime_test

U   = np.array(ds.U.values, dtype='float')
V   = np.array(ds.V.values, dtype='float')
Ug  = np.array(ds.Ug.values, dtype='float')
Vg  = np.array(ds.Vg.values, dtype='float')
TAx = np.array(ds.oceTAUX.values, dtype='float')
TAy = np.array(ds.oceTAUY.values, dtype='float')
dUdt = np.array(np.gradient(U, dt_forcing, axis=0), dtype='float')
dVdt = np.array(np.gradient(V, dt_forcing, axis=0), dtype='float')

n_Ug = (Ug-Ug.mean())/Ug.std()
n_Vg = (Vg-Vg.mean())/Vg.std()
n_TAx = (TAx-TAx.mean())/TAx.std()
n_TAy = (TAy-TAy.mean())/TAy.std()
n_U = (U-U.mean())/U.std()
n_V = (V-V.mean())/V.std()


train_data ={'dUdt' : dUdt[1:Ntime+1,:,:],
             'dVdt' : dUdt[1:Ntime+1,:,:],
             'U'    : U[:Ntime,:,:],
             'V'    : V[:Ntime,:,:],
             'TAx'  : TAx[:Ntime,:,:],
             'TAy'  : TAy[:Ntime,:,:],
             'features'   : np.stack([n_Ug[:Ntime,:,:],
                                      n_Vg[:Ntime,:,:],
                                      n_U[:Ntime,:,:],
                                      n_V[:Ntime,:,:],
                                    #   n_TAx[:Ntime,:,:],
                                    #   n_TAy[:Ntime,:,:]
                                      ], axis=1),
            }
test_data ={'dUdt' : dUdt[Ntime+1:,:,:],
             'dVdt' : dUdt[Ntime+1:,:,:],
             'U'    : U[Ntime:-1,:,:],
             'V'    : V[Ntime:-1,:,:],
             'TAx'  : TAx[Ntime:-1,:,:],
             'TAy'  : TAy[Ntime:-1,:,:],
             'features'   : np.stack([n_Ug[Ntime:-1,:,:],
                                      n_Vg[Ntime:-1,:,:],
                                      n_U[Ntime:-1,:,:],
                                      n_V[Ntime:-1,:,:],
                                    #   n_TAx[Ntime:-1,:,:],
                                    #   n_TAy[Ntime:-1,:,:]
                                      ], axis=1),
            }
print("Train data shape :",train_data['U'].shape)
print("Test data shape :",test_data['U'].shape)


fc = jnp.asarray(2*2*np.pi/86164*np.sin(ds.lat.values*np.pi/180))
K = jnp.asarray(jnp.exp(-10.)) # , dtype='float'


key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
mydissNN = DissipationNN(subkey, train_data['features'].shape[1])


myRHS = RHS(fc, K, mydissNN)

iter_train_data = dataloader(train_data, batch_size=BATCH_SIZE)
# test_iter_data = dataloader(train_data, batch_size=20)
if TRAIN:
    print('Now training ...')
    last_model, best_model, Lloss, Ltest = train(
                                                myRHS, 
                                                optim=optax.adam(LEARNING_RATE),
                                                iter_train_data=iter_train_data,
                                                test_data = test_data,
                                                print_every=PRINT_EVERY,
                                                maxstep=MAX_STEP
                                                )
    eqx.tree_serialise_leaves('./best_RHS.pt',best_model)   # <- saves the pytree
    
    fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
    epochs = np.arange(len(Lloss))
    ax.plot(epochs, Lloss, label='train')
    ax.plot(epochs, Ltest, label='test')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    print('done!')




trained_model = eqx.tree_deserialise_leaves('./best_RHS_5000epochs.pt',        # <- getting the saved PyTree 
                                            RHS(fc, K, mydissNN)    # <- here the call to RHS is just to get the structure
                                            )
train_loss = 0
N_batch = Ntime//BATCH_SIZE
for k in range(N_batch):
    batch_train = {key:array[k:k+BATCH_SIZE] for (key,array) in train_data.items() }
    train_loss += evaluate(trained_model, batch_train)
train_loss = train_loss / N_batch
test_loss = evaluate(trained_model, test_data)
print('best model loss:')
print(f'train loss = {train_loss}')
print(f'test loss = {test_loss}')

RHS_U_inference = trained_model(U[-2], V[-2], TAx[-2], TAy[-2], test_data['features'][-2])[0]
U_inf = U[-2] + dt*RHS_U_inference
# print(dUdt[-1])
# print(RHS_U_inference)
# print(U_inf)

# print(trained_model.dissipation.layer1.weight)
# print(trained_model.dissipation.layer1.bias)
# print(trained_model.dissipation.layer2.weight)
# print(trained_model.dissipation.layer2.bias)

fig, ax = plt.subplots(1,2,figsize = (10,5), constrained_layout=True,dpi=100)
ax[0].pcolormesh(dUdt[-1], vmin=-1e-5, vmax=1e-5)
ax[0].set_title('true dUdt (at last time)')
ax[1].pcolormesh(RHS_U_inference, vmin=-1e-5, vmax=1e-5)
ax[1].set_title('estimated dUdt (at last time)')


fig, ax = plt.subplots(1,2,figsize = (10,5), constrained_layout=True,dpi=100)
ax[0].pcolormesh(U[-1], vmin=-0.5, vmax=0.5)
ax[0].set_title('true U (at last time)')
ax[1].pcolormesh(U_inf, vmin=-0.5, vmax=0.5)
ax[1].set_title('estimated U (Euler) (at last time)')


Nsteps = 4*24
iinit = 0
features = np.stack([n_Ug, n_Vg, n_U, n_V,
                    #  n_TAx,
                    #  n_TAy
                     ], axis=1)[iinit:iinit+Nsteps]
forcing = TAx[iinit:iinit+Nsteps], TAy[iinit:iinit+Nsteps], features
trajU, trajV = Forward_Euler((U[iinit],V[iinit]), forcing=forcing, RHS=trained_model, dt=dt, Nsteps=Nsteps)


fig, ax = plt.subplots(1,2,figsize = (10,5), constrained_layout=True,dpi=100)
ax[0].pcolormesh(U[Nsteps-1], vmin=-0.5, vmax=0.5)
ax[0].set_title(f'true U at t={(Nsteps-1)*dt/3600}h')
ax[1].pcolormesh(trajU[Nsteps-1], vmin=-0.5, vmax=0.5)
ax[1].set_title(f'estimated U (Euler)at t={(Nsteps-1)*dt/3600}h')


# check divergence with time

errU = np.mean(np.abs( (U[iinit:iinit+Nsteps] - trajU)/U[iinit:iinit+Nsteps]), axis=(1,2)) *100
errV = np.mean(np.abs( (V[iinit:iinit+Nsteps] - trajV)/V[iinit:iinit+Nsteps]), axis=(1,2)) *100

RHSt = jax.vmap(trained_model)(U[iinit:iinit+Nsteps], 
                               V[iinit:iinit+Nsteps],
                               TAx[iinit:iinit+Nsteps], 
                               TAy[iinit:iinit+Nsteps], 
                               features)
errdUdt = np.mean(np.abs( (dUdt[iinit+1:iinit+Nsteps+1] - RHSt[0])/dUdt[iinit+1:iinit+Nsteps+1]), axis=(1,2)) *100
errdVdt = np.mean(np.abs( (dVdt[iinit+1:iinit+Nsteps+1] - RHSt[1])/dVdt[iinit+1:iinit+Nsteps+1]), axis=(1,2)) *100
# evolution of error with number of timestep
fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
ax.plot(errU, label='errU (%)')
ax.plot(errV, label='errV (%)')
ax.set_ylim([0,100])
ax.set_ylabel('error on U')
ax.legend()
ax.set_xlabel('hours')

fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
ax.plot(errdUdt, label='errdUdt (%)')
ax.plot(errdVdt, label='errdVdt (%)')
ax.set_ylim([0,100])
ax.set_ylabel('error on dUdt')
ax.legend()
ax.set_xlabel('hours')




# evolution of <dUdt>
fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
ax.plot(np.arange(iinit+1,iinit+Nsteps+1), dUdt[iinit+1:iinit+Nsteps+1].mean(axis=(1,2)), label='true <dUdt>xy')
ax.plot(np.arange(iinit+1,iinit+Nsteps+1), RHSt[0].mean(axis=(1,2)), label='estimated <dUdt>xy')
ax.set_ylim([-0.00002,0.00002])
ax.legend()
ax.set_xlabel('hours')
# evolution of <U>
fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
ax.plot(U[iinit:iinit+Nsteps].mean(axis=(1,2)), label='true <U>xy')
ax.plot(trajU.mean(axis=(1,2)), label='estimated <U>xy')
ax.set_ylim([-0.3,0.3])
ax.legend()
ax.set_xlabel('hours')

fig, ax = plt.subplots(1,1,figsize = (5,5), constrained_layout=True,dpi=100)
s = ax.pcolormesh( trajU[Nsteps-1] - U[Nsteps-1], cmap='bwr',vmin=-0.5, vmax=0.5)
plt.colorbar(s,ax=ax)
ax.set_title(f'delta U at t={(Nsteps-1)*dt/3600}h')



plt.show()

