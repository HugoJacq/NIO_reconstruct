
"""
Goal: 
    Estimate ageostrophic current, from a simple slab model. Given wind stress, geostrophy (sst ?)


Context:

    the slab equation in complex notation is:

        dC/dt = -i.fc.C + K0.Tau - dissipation              (1)

    with C = U+i.V
         Tau = Taux + i.Tauy
         K0 = real
         fc = Coriolis frequency
         
    the dissipation term is usually parametrized as a Rayleigh damping term rC with r a constant.
    
Our approach:

    we think that using a Rayleigh damping term is very restrictive. Seeking for more a more expressive term, 
    we try to model it using a neural network.
    
    either the full dissipation as a NN(Ug, U, Tau), or just the r as a constant, or the r(Ug, U, Tau) as a function of features 

This script:

    We aim to have a dissipation term that allow for a prediction of the RHS of (1), then integrate this one or multiple time to get 
        a surface current trajectory.
        
        Loss = || target - dissipation ||
        
            with target = dCdt + RHS_dynamic
"""
# regular modules import
import xarray as xr
import numpy as np
import jax
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import os 
import sys
sys.path.insert(0, '../../src')
# jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
# jax.config.update("jax_enable_x64", True)

# my modules imports
from training import data_maker, batch_loader, train, normalize_batch, features_maker
from NNmodels import *
from constants import oneday, distance_1deg_equator

# NN hyper parameters
TRAIN           = False      # run the training and save best model
SAVE            = False     # save model and figures
LEARNING_RATE   = 1e-3      # initial learning rate for optimizer
MAX_STEP        = 200     # number of epochs
PRINT_EVERY     = 10        # print infos every 'PRINT_EVERY' epochs
BATCH_SIZE      = -1       # size of a batch (time), set to -1 for no batching
SEED            = 5678      # for reproductibility
features_names  = ['U','V','gradxU']   # what features to use in the NN

PLOTTING        = True
TIME_INTEG      = True

# Defining data
Ntests      = 10*24     # number of hours for test (over the data)
Nsize       = 128       # size of the domain, in nx ny
dt_forcing  = 3600      # time step of forcing
dt          = 3600.     # time step for Euler integration
dx          = 0.1       # X data resolution in ° 
dy          = 0.1       # Y data resolution in ° 
K0          = np.exp(-8.)      # initial K0


filename = ['../../data_regrid/croco_1h_inst_surf_2005-01-01-2005-01-31_0.1deg_conservative.nc',
            # '../../data_regrid/croco_1h_inst_surf_2005-02-01-2005-02-28_0.1deg_conservative.nc',
            # '../../data_regrid/croco_1h_inst_surf_2005-03-01-2005-03-31_0.1deg_conservative.nc',
            # "../../data_regrid/croco_1h_inst_surf_2005-04-01-2005-04-30_0.1deg_conservative.nc"
            ]
# prepare data
ds = xr.open_mfdataset(filename)
ds = ds.isel(lon=slice(-Nsize-1,-1),lat=slice(-Nsize-1,-1)) # <- this could be centered on a physical location
ds = ds.rename({'oceTAUX':'TAx', 'oceTAUY':'TAy'})
fc = np.asarray(2*2*np.pi/86164*np.sin(ds.lat.values*np.pi/180))

# intialize
my_dynamic_RHS = RHS_dynamic(fc, K0)
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)


L_my_diss = {
            # "DissipationRayleigh":DissipationRayleigh(),
            # "DissipationRayleigh_NNlinear_1":DissipationRayleigh_NNlinear(subkey, Nfeatures=len(features_names), width=1),
            # "DissipationRayleigh_NNlinear_1024":DissipationRayleigh_NNlinear(subkey, Nfeatures=len(features_names), width=1024),
            # "DissipationRayleigh_MLP":DissipationRayleigh_MLP(subkey, Nfeatures=len(features_names), width=1024),
            # "DissipationMLP":DissipationMLP(subkey, Nfeatures=len(features_names)),
            "DissipationCNN":DissipationCNN(subkey, Nfeatures=len(features_names)),
            }

# make test and train datasets
train_set, test_set = data_maker(ds, 
                                 Ntests, 
                                 features_names, 
                                 my_dynamic_RHS, 
                                 dx=dx*distance_1deg_equator, dy=dy*distance_1deg_equator)
iter_train_data = batch_loader(train_set,
                               batch_size=BATCH_SIZE)

for namediss,my_diss in L_my_diss.items():
    # namediss = type(my_diss).__name__
    path_save = namediss+'/'
    os.system(f'mkdir -p {path_save}')
    
    print('')
    print(f'-> working on {namediss}')
    
    
    # run the training
    if TRAIN:
        print('* training ...')
        model, best_model, Train_loss, Test_loss = train(
                                                    diss_model          = my_diss,
                                                    optim               = optax.adam(LEARNING_RATE), # optax.adam(LEARNING_RATE), #optax.lbfgs(LEARNING_RATE),
                                                    iter_train_data     = iter_train_data,
                                                    train_data          = train_set,
                                                    test_data           = test_set,
                                                    maxstep             = MAX_STEP,
                                                    print_every         = PRINT_EVERY,
                                                    )
        if SAVE:
            eqx.tree_serialise_leaves(path_save+'best_diss.pt',best_model)   # <- saves the pytree

        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
        epochs = np.arange(len(Train_loss))
        epochs_test = np.arange(len(Test_loss))*PRINT_EVERY
        ax.plot(epochs, Train_loss, label='train')
        ax.plot(epochs_test, Test_loss, label='test')
        ax.set_xlabel('epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        if SAVE:
            fig.savefig(path_save+'Loss.png')
        print('done!')

    trained_diss = eqx.tree_deserialise_leaves(path_save+'best_diss.pt',        # <- getting the saved PyTree 
                                                my_diss    # <- here the call is just to get the structure
                                                )
    # print('target mean, std U:', trained_diss.RENORMmean[0], trained_diss.RENORMstd[0])
    # print('target mean, std V:', trained_diss.RENORMmean[1], trained_diss.RENORMstd[1])

    # print('final R value is',trained_diss.layers[0].R)
    if PLOTTING:
        print('* Plotting')
        # TEST DATA
        my_test_data = normalize_batch(test_set, deep_copy=True)
        mydiss_undim = jax.vmap(trained_diss)(my_test_data['features'])
        mydiss_dim = mydiss_undim*trained_diss.RENORMstd[:,np.newaxis, np.newaxis] + trained_diss.RENORMmean[:,np.newaxis, np.newaxis]
        xtime = np.arange(len(ds.time)-Ntests, len(ds.time)-1 )*dt_forcing/oneday

        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
        ax.plot(xtime, mydiss_dim[:,0].mean(axis=(1,2)), label='U diss estimate', c='b')
        ax.plot(xtime, test_set['target'][:,0].mean(axis=(1,2)), label='U target', c='b', alpha=0.5)
        ax.plot(xtime, mydiss_dim[:,1].mean(axis=(1,2)), label='V diss estimate', c='orange')
        ax.plot(xtime, test_set['target'][:,1].mean(axis=(1,2)), label='Vtarget', c='orange', alpha=0.5)
        ax.set_xlabel('time (days)')
        ax.set_ylabel('dissipation')
        ax.set_title('test_data')
        ax.legend()
        if SAVE:
            fig.savefig(path_save+'test_data_dim.png')
            
        # TRAIN DATA
        my_train_data = normalize_batch(train_set, deep_copy=True)
        mydiss_undim = jax.vmap(trained_diss)(my_train_data['features'])
        mydiss_dim = mydiss_undim*trained_diss.RENORMstd[:,np.newaxis, np.newaxis] + trained_diss.RENORMmean[:,np.newaxis, np.newaxis]
        xtime = np.arange(0, len(ds.time)-Ntests )*dt_forcing/oneday

        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
        ax.plot(xtime, mydiss_dim[:,0].mean(axis=(1,2)), label='U diss estimate', c='b')
        ax.plot(xtime, train_set['target'][:,0].mean(axis=(1,2)), label='U target', c='b', alpha=0.5)
        ax.plot(xtime, mydiss_dim[:,1].mean(axis=(1,2)), label='V diss estimate', c='orange')
        ax.plot(xtime, train_set['target'][:,1].mean(axis=(1,2)), label='Vtarget', c='orange', alpha=0.5)
        ax.set_xlabel('time (days)')
        ax.set_ylabel('dissipation')
        ax.set_title('train_data')
        ax.legend()    
        if SAVE:
            fig.savefig(path_save+'train_data_dim.png')

        # non dim data
        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
        ax.plot(xtime, mydiss_undim[:,0].mean(axis=(1,2)), label='estimated U', c='b')
        ax.plot(xtime, my_train_data['target'][:,0].mean(axis=(1,2)), label='U target', c='b', alpha=0.5)
        ax.plot(xtime, mydiss_undim[:,1].mean(axis=(1,2)), label='estimated V', c='orange')
        ax.plot(xtime, my_train_data['target'][:,1].mean(axis=(1,2)), label='V target', c='orange', alpha=0.5)
        ax.set_title('non dim, train_data')
        if SAVE:
            fig.savefig(path_save+'train_data_undim.png')

        # terms of U budget
        Coriolis = fc.mean()*ds.V.isel(time=slice(0,len(ds.time)-Ntests)).mean(axis=(1,2)).values
        Stress = K0*ds.TAx.isel(time=slice(0,len(ds.time)-Ntests)).mean(axis=(1,2)).values
        true_diss = train_set['target'][:,0].mean(axis=(1,2))
        est_diss = mydiss_dim[:,0].mean(axis=(1,2))
        true_dudt = np.gradient(ds.U.isel(time=slice(0,len(ds.time)-Ntests)), 3600., axis=0).mean(axis=(1,2))
        est_dudt = Coriolis + Stress + est_diss
        xtime = np.arange(0,len(Coriolis))*dt_forcing/oneday

        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
        ax.plot(xtime, Coriolis, label='Coriolis', c='orange')
        ax.plot(xtime, Stress, label='K0*tau',c='g')
        ax.plot(xtime, true_diss, label='true diss', c='b')
        ax.plot(xtime, est_diss, label='estimated diss', c='cyan')
        ax.plot(xtime, true_dudt, label='true dUdt',c='k')
        ax.plot(xtime, est_dudt, label='estimated dUdt',c='k',ls='--')
        ax.set_xlabel('days')
        ax.set_ylabel('m/s2')
        ax.set_title('U budget, train period')
        ax.legend()
        if SAVE:
            fig.savefig(path_save+'U_budget.png')

    if TIME_INTEG:
        print('* Time integration')
        Nhours = 24
        Nsteps = Nhours*60
        dt = 60.
        
        # features = np.stack([ds.U.values, ds.V.values], axis=1)
        features = features_maker(ds, features_names, dx, dy, out_axis=1, out_dtype='float')
        forcing = ds.TAx.values, ds.TAy.values, features
        U,V = Forward_Euler(X0=(0.,0.), RHS_dyn=my_dynamic_RHS, diss_model=trained_diss, forcing=forcing, dt=dt, dt_forcing=dt_forcing, Nsteps=Nsteps)
        xtime = np.arange(0, Nsteps*dt, dt)/onehour
        
        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
        ax.plot(xtime, U.mean(axis=(1,2)), label='estimated')
        ax.plot(np.arange(0,Nhours), ds.U.isel(time=slice(0,Nhours)).mean(axis=(1,2)).values, label='true')
        ax.set_xlabel('hour')
        ax.set_title('integrated (Euler) vs truth')
        ax.set_ylabel('U')
        if SAVE:
            fig.savefig(path_save+'U_traj_Euler.png')
        
plt.show()