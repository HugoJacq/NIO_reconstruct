
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
        
    This script is offline training as we compare the target and the ouput of the NN at every timestep.
    The input data is a collection of images, and we compute a loss for each of them. 
        
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
jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
# jax.config.update("jax_enable_x64", True)

# my modules imports
from preprocessing import data_maker, features_maker
from training import batch_loader, train, normalize_batch
from NNmodels import *
from constants import oneday, distance_1deg_equator

#====================================================================================================
# PARAMETERS
#====================================================================================================

# NN hyper parameters
TRAIN           = False      # run the training and save best model
SAVE            = True     # save model and figures
LEARNING_RATE   = 1e-3      
MAX_STEP        = 300        # number of epochs
PRINT_EVERY     = 10        # print infos every 'PRINT_EVERY' epochs
BATCH_SIZE      = 200      # size of a batch (time), set to -1 for no batching
SEED            = 5678      # for reproductibility
features_names  = ['U','V','TAx','TAy']   # what features to use in the NN
mymode          = 'NN_only' # NN_only, hybrid

# evolving learning rate (linear)
lr_init = 1e-3      # initial lr
lr_end = 1e-3       # end lr
tr_steps = 30       # how many steps to go from lr_init to lr_end
tr_start = 100      # how many iteration before changing lr

PLOTTING        = True
TIME_INTEG      = True

# Defining data
# Ntests      = 10*24       # number of hours for test (over the data)
test_ratio  = 20             # % of hours for test (over the data)
Nsize       = 128           # size of the domain, in nx ny
dt_forcing  = 3600          # time step of forcing
dt          = 3600.         # time step for Euler integration
dx          = 0.1           # X data resolution in ° 
dy          = 0.1           # Y data resolution in ° 
K0          = 1.           # initial K0


filename = ['../../data_regrid/croco_1h_inst_surf_2005-01-01-2005-01-31_0.1deg_conservative.nc',
            '../../data_regrid/croco_1h_inst_surf_2005-02-01-2005-02-28_0.1deg_conservative.nc',
            '../../data_regrid/croco_1h_inst_surf_2005-03-01-2005-03-31_0.1deg_conservative.nc',
            "../../data_regrid/croco_1h_inst_surf_2005-04-01-2005-04-30_0.1deg_conservative.nc"
            ]
# prepare data
ds = xr.open_mfdataset(filename)
ds = ds.isel(lon=slice(-Nsize-1,-1),lat=slice(-Nsize-1,-1)) # <- this could be centered on a physical location
ds = ds.rename({'oceTAUX':'TAx', 'oceTAUY':'TAy'})
fc = np.asarray(2*2*np.pi/86164*np.sin(ds.lat.values*np.pi/180))

if mymode=='NN_only': # temporaire
    K0 = -10.
# intialize
my_dynamic_RHS = RHS_dynamic(fc, K0)
my_coriolis_term = Coriolis(fc)
my_stress_term = Stress(K0)
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
L_my_RHS_hybrid = {
            # "RHS_dissRayleigh": RHS_turb(my_stress_term, DissipationRayleigh()),
            "RHS_dissCNN": RHS_turb(my_stress_term,  DissipationCNN(subkey, Nfeatures=len(features_names)))
            }

L_no_param_RHS = {"RHS_dissRayleigh":my_coriolis_term,
                  "RHS_dissCNN":my_coriolis_term,
                  
                  "DissipationCNN":my_dynamic_RHS}
#====================================================================================================
# END PARAMETERS
#====================================================================================================
Ntests = test_ratio*len(ds.time)//100

if mymode=='NN_only':
    L_models = L_my_diss
elif mymode=='hybrid':
    L_models = L_my_RHS_hybrid
else:
    raise Exception(f'You want to use the mode {mymode} but it is not recognized')


learning_rate_fn = optax.linear_schedule(lr_init, lr_end, tr_start)

# create folder for each set of features
name_base_folder = 'features'+''.join('_'+variable for variable in features_names)+'/'
os.system(f'mkdir -p {name_base_folder}')

for namemodel, my_model in L_models.items():
    
    # make test and train datasets
    train_set, test_set = data_maker(ds, 
                                    Ntests, 
                                    features_names, 
                                    L_no_param_RHS[namemodel], 
                                    dx=dx*distance_1deg_equator, 
                                    dy=dy*distance_1deg_equator,
                                    dt_forcing=dt_forcing,
                                    mode=mymode)
    iter_train_data = batch_loader(train_set,
                                batch_size=BATCH_SIZE)
    
    
    # namediss = type(my_diss).__name__
    path_save = name_base_folder+namemodel+'/'
    os.system(f'mkdir -p {path_save}')
    
    print('')
    print(f'-> working on {namemodel}')
    
    
    # run the training
    if TRAIN:
        print('* training ...')
        model, best_model, Train_loss, Test_loss = train(
                                                    the_model           = my_model,
                                                    optim               = optax.adam(learning_rate_fn), # optax.adam(LEARNING_RATE), #optax.lbfgs(LEARNING_RATE),
                                                    iter_train_data     = iter_train_data,
                                                    train_data          = train_set,
                                                    test_data           = test_set,
                                                    maxstep             = MAX_STEP,
                                                    print_every         = PRINT_EVERY,
                                                    mode                = mymode,
                                                    )
        if SAVE:
            eqx.tree_serialise_leaves(path_save+'best_model.pt',best_model)   # <- saves the pytree

        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
        epochs = np.arange(len(Train_loss))
        epochs_test = np.arange(len(Test_loss))*PRINT_EVERY
        ax.plot(epochs, Train_loss, label='train')
        ax.plot(epochs_test, Test_loss, label='test')
        ax.set_xlabel('epochs')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()
        if SAVE:
            fig.savefig(path_save+'Loss.png')
        print('done!')

    trained_model = eqx.tree_deserialise_leaves(path_save+'best_model.pt',   # <- getting the saved PyTree 
                                                my_model                    # <- here the call is just to get the structure
                                                )
    print('target mean, std U:', trained_model.RENORMmean[0], trained_model.RENORMstd[0])
    print('target mean, std V:', trained_model.RENORMmean[1], trained_model.RENORMstd[1])

    # print('final R value is',trained_model.layers[0].R)
    if PLOTTING:
        print('* Plotting')
        
        if mymode=='NN_only':
        
            # TEST DATA
            my_test_data = normalize_batch(test_set, deep_copy=True)
            mydiss_undim = jax.vmap(trained_model)(my_test_data['features'])
            mydiss_dim_test = mydiss_undim*trained_model.RENORMstd[:,np.newaxis, np.newaxis] + trained_model.RENORMmean[:,np.newaxis, np.newaxis]
            xtime = np.arange(len(ds.time)-Ntests, len(ds.time)-1 )*dt_forcing/oneday

            fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
            ax.plot(xtime, mydiss_dim_test[:,0].mean(axis=(1,2)), label='U diss estimate', c='b')
            ax.plot(xtime, test_set['target'][:,0].mean(axis=(1,2)), label='U target', c='b', alpha=0.5)
            ax.plot(xtime, mydiss_dim_test[:,1].mean(axis=(1,2)), label='V diss estimate', c='orange')
            ax.plot(xtime, test_set['target'][:,1].mean(axis=(1,2)), label='Vtarget', c='orange', alpha=0.5)
            ax.set_xlabel('time (days)')
            ax.set_ylabel('dissipation')
            ax.set_title('test_data')
            ax.legend()
            if SAVE:
                fig.savefig(path_save+'test_data_dim.png')
                
            # TRAIN DATA
            my_train_data = normalize_batch(train_set, deep_copy=True)
            mydiss_undim = jax.vmap(trained_model)(my_train_data['features'])
            mydiss_dim_train = mydiss_undim*trained_model.RENORMstd[:,np.newaxis, np.newaxis] + trained_model.RENORMmean[:,np.newaxis, np.newaxis]
            xtime = np.arange(0, len(ds.time)-Ntests )*dt_forcing/oneday

            fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
            ax.plot(xtime, mydiss_dim_train[:,0].mean(axis=(1,2)), label='U diss estimate', c='b')
            ax.plot(xtime, train_set['target'][:,0].mean(axis=(1,2)), label='U target', c='b', alpha=0.5)
            ax.plot(xtime, mydiss_dim_train[:,1].mean(axis=(1,2)), label='V diss estimate', c='orange')
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

            # terms of U budget on train data
            coriolis = fc.mean()*ds.V.isel(time=slice(0,len(ds.time)-Ntests)).mean(axis=(1,2)).values
            stress = np.exp(K0)*ds.TAx.isel(time=slice(0,len(ds.time)-Ntests)).mean(axis=(1,2)).values
            true_diss = train_set['target'][:,0].mean(axis=(1,2))
            est_diss = mydiss_dim_train[:,0].mean(axis=(1,2))
            true_dudt = np.gradient(ds.U.isel(time=slice(0,len(ds.time)-Ntests)), dt_forcing, axis=0).mean(axis=(1,2))
            est_dudt = coriolis + stress + est_diss
            xtime = np.arange(0,len(coriolis))*dt_forcing/oneday

            fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
            ax.plot(xtime, coriolis, label='Coriolis', c='orange')
            ax.plot(xtime, stress, label='K0*tau',c='g')
            ax.plot(xtime, true_diss, label='true diss', c='b')
            ax.plot(xtime, est_diss, label='estimated diss', c='cyan')
            ax.plot(xtime, true_dudt, label='true dUdt',c='k')
            ax.plot(xtime, est_dudt, label='estimated dUdt',c='k',ls='--')
            ax.set_xlabel('days')
            ax.set_ylabel('m/s2')
            ax.set_title('U budget, train period')
            ax.legend()
            if SAVE:
                fig.savefig(path_save+'U_budget_train.png')
                
            # terms of U budget on test data
            coriolis = fc.mean()*ds.V.isel(time=slice(len(ds.time)-Ntests, -1)).mean(axis=(1,2)).values
            stress = np.exp(K0)*ds.TAx.isel(time=slice(len(ds.time)-Ntests, -1)).mean(axis=(1,2)).values
            true_diss = test_set['target'][:,0].mean(axis=(1,2))
            est_diss = mydiss_dim_test[:,0].mean(axis=(1,2))
            true_dudt = np.gradient(ds.U.isel(time=slice(len(ds.time)-Ntests, -1)), dt_forcing, axis=0).mean(axis=(1,2))
            est_dudt = coriolis + stress + est_diss
            xtime = np.arange(0,len(coriolis))*dt_forcing/oneday

            fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
            ax.plot(xtime, coriolis, label='Coriolis', c='orange')
            ax.plot(xtime, stress, label='K0*tau',c='g')
            ax.plot(xtime, true_diss, label='true diss', c='b')
            ax.plot(xtime, est_diss, label='estimated diss', c='cyan')
            ax.plot(xtime, true_dudt, label='true dUdt',c='k')
            ax.plot(xtime, est_dudt, label='estimated dUdt',c='k',ls='--')
            ax.set_xlabel('days')
            ax.set_ylabel('m/s2')
            ax.set_title('U budget, test period')
            ax.legend()
            if SAVE:
                fig.savefig(path_save+'U_budget_test.png')

            if TIME_INTEG:
                print('* Time integration')
                Nhours = 24
                Nsteps = Nhours*60
                dt = 60.
                
                # features = np.stack([ds.U.values, ds.V.values], axis=1)
                features = features_maker(ds, features_names, dx, dy, out_axis=1, out_dtype='float')
                forcing = ds.TAx.values, ds.TAy.values, features
                U,V = Forward_Euler(X0=(0.,0.), RHS_dyn=my_dynamic_RHS, diss_model=trained_model, forcing=forcing, dt=dt, dt_forcing=dt_forcing, Nsteps=Nsteps)
                xtime = np.arange(0, Nsteps*dt, dt)/onehour
                
                fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
                ax.plot(xtime, U.mean(axis=(1,2)), label='estimated')
                ax.plot(np.arange(0,Nhours), ds.U.isel(time=slice(0,Nhours)).mean(axis=(1,2)).values, label='true')
                ax.set_xlabel('hour')
                ax.set_title('integrated (Euler) vs truth')
                ax.set_ylabel('U')
                if SAVE:
                    fig.savefig(path_save+'U_traj_Euler.png')
                    
        elif mymode=='hybrid':
            
            print(f'K0 value is {trained_model.stress.K}')
            
             # TEST DATA
            my_test_data = normalize_batch(test_set, deep_copy=True)
            # myRHS_undim_test = jax.vmap(trained_model)(my_test_data['features'], my_test_data['forcing'])
            
            U,V = my_test_data['features'][:,0], my_test_data['features'][:,1]
            TAx,TAy = my_test_data['forcing'][:,0], my_test_data['forcing'][:,1] 
            stress_undim_test = jax.vmap(trained_model.stress)(TAx,TAy)
            stress_dim_test = stress_undim_test*trained_model.RENORMstd[:,np.newaxis, np.newaxis] + trained_model.RENORMmean[:,np.newaxis, np.newaxis]
            coriolis_test = np.stack([fc*V, -fc*U], axis=1)
            diss_undim_test = jax.vmap(trained_model.dissipationNN)(my_test_data['features'])
            diss_dim_test = diss_undim_test*trained_model.RENORMstd[:,np.newaxis, np.newaxis] + trained_model.RENORMmean[:,np.newaxis, np.newaxis]
            myRHS_test = stress_dim_test + diss_dim_test + coriolis_test
            xtime = np.arange(len(ds.time)-Ntests, len(ds.time)-1 )*dt_forcing/oneday
            
            fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
            ax.plot(xtime, myRHS_test[:,0].mean(axis=(1,2)), label='RHS U estimate', c='b')
            ax.plot(xtime, (test_set['target']+ coriolis_test)[:,0].mean(axis=(1,2)), label='dUdt target', c='b', alpha=0.5)
            ax.plot(xtime, myRHS_test[:,1].mean(axis=(1,2)), label='RHS V estimate', c='orange')
            ax.plot(xtime, (test_set['target']+ coriolis_test)[:,1].mean(axis=(1,2)), label='dVdt target', c='orange', alpha=0.5)
            ax.set_xlabel('time (days)')
            ax.set_ylabel('RHS')
            ax.set_title('test_data')
            ax.set_ylim([-1e-4,1e-4])
            ax.legend()
            if SAVE:
                fig.savefig(path_save+'test_data_dim.png')
               
           # TRAIN DATA
            my_train_data = normalize_batch(train_set, deep_copy=True)
            
            U,V = my_train_data['features'][:,0], my_train_data['features'][:,1]
            TAx,TAy = my_train_data['forcing'][:,0], my_train_data['forcing'][:,1] 
            stress_undim_train = jax.vmap(trained_model.stress)(TAx,TAy)
            stress_dim_train = stress_undim_train*trained_model.RENORMstd[:,np.newaxis, np.newaxis] + trained_model.RENORMmean[:,np.newaxis, np.newaxis]
            coriolis_train = np.stack([fc*V, -fc*U], axis=1)
            diss_undim_train = jax.vmap(trained_model.dissipationNN)(my_train_data['features'])
            diss_dim_train = diss_undim_train*trained_model.RENORMstd[:,np.newaxis, np.newaxis] + trained_model.RENORMmean[:,np.newaxis, np.newaxis]
            myRHS_train = coriolis_train + stress_dim_train + diss_dim_train
            xtime = np.arange(0, len(ds.time)-Ntests )*dt_forcing/oneday

            fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
            ax.plot(xtime, myRHS_train[:,0].mean(axis=(1,2)), label='RHS U estimate', c='b')
            ax.plot(xtime, (train_set['target']+coriolis_train)[:,0].mean(axis=(1,2)), label='dUdt target', c='b', alpha=0.5)
            ax.plot(xtime, myRHS_train[:,1].mean(axis=(1,2)), label='RHS V estimate', c='orange')
            ax.plot(xtime, (train_set['target']+coriolis_train)[:,1].mean(axis=(1,2)), label='dVdt target', c='orange', alpha=0.5)
            ax.set_xlabel('time (days)')
            ax.set_ylabel('RHS')
            ax.set_title('train_data')
            ax.set_ylim([-1e-4,1e-4])
            ax.legend()    
            if SAVE:
                fig.savefig(path_save+'train_data_dim.png') 
                  
            # non dim data
            n_stress_train = jax.vmap(trained_model.stress)(my_train_data['forcing'][:,0], 
                                                            my_train_data['forcing'][:,1] )
            n_target_train = my_train_data['target']
            n_diss_train = jax.vmap(trained_model.dissipationNN)(my_train_data['features'])
            xtime = np.arange(0, len(ds.time)-Ntests )*dt_forcing/oneday
            
            fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
            ax.plot(xtime, n_stress_train[:,0].mean(axis=(1,2)), label='estimated stress', c='b')
            ax.plot(xtime, n_diss_train[:,0].mean(axis=(1,2)), label='estimated diss', c='g')
            ax.plot(xtime, (n_diss_train+n_stress_train)[:,0].mean(axis=(1,2)), label='estimated sum', c='k', ls='--')
            ax.plot(xtime, n_target_train[:,0].mean(axis=(1,2)), label='target', c='k')
            ax.set_title('non dim, train_data')
            ax.legend()
            if SAVE:
                fig.savefig(path_save+'train_data_undim.png')            
                
            # terms of U budget on train data
            coriolis = fc.mean()*ds.V.isel(time=slice(0,len(ds.time)-Ntests)).mean(axis=(1,2)).values
            stress_undim = jax.vmap(trained_model.stress)(my_train_data['forcing'][:,0],
                                                          my_train_data['forcing'][:,1])
            stress_dim = (stress_undim*trained_model.RENORMstd[:,np.newaxis, np.newaxis] + trained_model.RENORMmean[:,np.newaxis, np.newaxis]
                            )[:,0].mean(axis=(1,2))
            est_diss_undim = jax.vmap(trained_model.dissipationNN)(my_train_data['features']) 
            est_diss_dim = (est_diss_undim*trained_model.RENORMstd[:,np.newaxis, np.newaxis] + trained_model.RENORMmean[:,np.newaxis, np.newaxis]
                            )[:,0].mean(axis=(1,2))
            true_dudt = np.gradient(ds.U.isel(time=slice(0,len(ds.time)-Ntests)), dt_forcing, axis=0).mean(axis=(1,2))
            est_dudt = coriolis + stress_dim + est_diss_dim
            xtime = np.arange(0,len(coriolis))*dt_forcing/oneday

            fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
            ax.plot(xtime, coriolis, label='Coriolis', c='orange')
            ax.plot(xtime, stress_dim, label='K0*tau',c='g')
            ax.plot(xtime, est_diss_dim, label='estimated diss', c='cyan')
            ax.plot(xtime, true_dudt, label='true dUdt',c='k')
            ax.plot(xtime, est_dudt, label='estimated dUdt',c='k',ls='--')
            ax.set_xlabel('days')
            ax.set_ylabel('m/s2')
            ax.set_title('U budget, train period')
            ax.legend()
            if SAVE:
                fig.savefig(path_save+'U_budget_train.png')
                
            # terms of U budget on test data
            coriolis = fc.mean()*ds.V.isel(time=slice(len(ds.time)-Ntests,-1)).mean(axis=(1,2)).values
            stress_undim = jax.vmap(trained_model.stress)(my_test_data['forcing'][:,0],
                                                          my_test_data['forcing'][:,1])
            stress_dim = (stress_undim*trained_model.RENORMstd[:,np.newaxis, np.newaxis] + trained_model.RENORMmean[:,np.newaxis, np.newaxis]
                            )[:,0].mean(axis=(1,2))
            est_diss_undim = jax.vmap(trained_model.dissipationNN)(my_test_data['features']) 
            est_diss_dim = (est_diss_undim*trained_model.RENORMstd[:,np.newaxis, np.newaxis] + trained_model.RENORMmean[:,np.newaxis, np.newaxis]
                            )[:,0].mean(axis=(1,2))
            true_dudt = np.gradient(ds.U.isel(time=slice(len(ds.time)-Ntests,-1)), dt_forcing, axis=0).mean(axis=(1,2))
            est_dudt = coriolis + stress_dim + est_diss_dim
            xtime = np.arange(0,len(coriolis))*dt_forcing/oneday

            fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
            ax.plot(xtime, coriolis, label='Coriolis', c='orange')
            ax.plot(xtime, stress_dim, label='K0*tau',c='g')
            ax.plot(xtime, est_diss_dim, label='estimated diss', c='cyan')
            ax.plot(xtime, true_dudt, label='true dUdt',c='k')
            ax.plot(xtime, est_dudt, label='estimated dUdt',c='k',ls='--')
            ax.set_xlabel('days')
            ax.set_ylabel('m/s2')
            ax.set_title('U budget, test period')
            ax.legend()
            if SAVE:
                fig.savefig(path_save+'U_budget_test.png')

            
            if TIME_INTEG:
                print('* Time integration')
                Nhours = 24
                Nsteps = Nhours*60
                dt = 60.
                
                
                """
                To do: adapt 'Forward_Euler' à la nouvelle architecture pour pouvoir intégrer en temps la solution et comparer les trajectoires
                
                """
                
                # features = np.stack([ds.U.values, ds.V.values], axis=1)
                features = features_maker(ds, features_names, dx, dy, out_axis=1, out_dtype='float')
                forcing = ds.TAx.values, ds.TAy.values, features
                U,V = Forward_Euler(X0=(0.,0.), RHS_dyn=my_dynamic_RHS, diss_model=trained_model, forcing=forcing, dt=dt, dt_forcing=dt_forcing, Nsteps=Nsteps)
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