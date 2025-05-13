"""
Context:
    the slab equation in complex notation is:

        dC/dt = -i.fc.C + K0.Tau - dissipation              (1)

    with C = U+i.V
         Tau = Taux + i.Tauy
         K0 = real
         fc = Coriolis frequency
         
    the dissipation term is usually parametrized as a Rayleigh damping term rC with r a constant.

This script:

    We compare several parametrization of the dissipation term, physic-based or with a neural network, 
    and also different learning strategies (offline, online).
    
The question to be answered:

    Can a Neural Network improve the reconstruction of the ageostrophic current ? 
    
subquestions:
    Is the dissipation term more expressive ? Does this lead to better reconstructed currents ?

"""

# regular modules import
import pickle
import pprint
from re import L
import xarray as xr
import numpy as np
import jax
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
sys.path.insert(0, '../../src')
# jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
jax.config.update("jax_enable_x64", True)

# my imports
from train_preprocessing import data_maker, batch_loader, normalize_batch
from train_functions import train, my_partition, vmap_loss
from time_integration import Integration_Euler
from constants import oneday, onehour, distance_1deg_equator
from models_definition import RHS, Dissipation_Rayleigh, Stress_term, Coriolis_term
#====================================================================================================
# PARAMETERS
#====================================================================================================

TRAIN           = True      # run the training and save best model
SAVE            = True      # save model and figures
PLOTTING        = True      # plot figures

MAX_STEP        = 100           # number of epochs
PRINT_EVERY     = 10            # print infos every 'PRINT_EVERY' epochs
SEED            = 5678          # for reproductibility
features_names  = []            # what features to use in the NN (U,V in by default)
forcing_names   = []            # U,V,TAx,TAy already in by default

BATCH_SIZE      = -1          # size of a batch (time), set to -1 for no batching

dt_Euler    = 60.           # secondes
N_integration_steps = 1         # 1 for offline, more for online
N_integration_steps_verif = 1  # number of time step to integrate during in use cases of the model

# Defining data
test_ratio  = 20            # % of hours for test (over the data)
mydtype     = 'float'       # type of data
Nsize       = 128           # size of the domain, in nx ny
dt_forcing  = 3600.         # time step of forcing
dx          = 0.1           # X data resolution in ° 
dy          = 0.1           # Y data resolution in ° 


K0          = -10.5           # initial K0
R           = -10.5           # initial R
#====================================================================================================

#====================================================================================================

# create folder for each set of features
name_base_folder = 'features'+''.join('_'+variable for variable in features_names)+'/'
os.system(f'mkdir -p {name_base_folder}')

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
Ntests = test_ratio*len(ds.time)//100 # how many instants used for test
Ntrain = len(ds.time) - Ntests

# warnings
if BATCH_SIZE<0:
    BATCH_SIZE = len(ds.time) - Ntests - 2
if N_integration_steps > BATCH_SIZE and BATCH_SIZE>0:
    print(f'You have chosen to do online training but the number of integration step ({N_integration_steps}) is greater than the batch_size ({BATCH_SIZE})')
    print(f'N_integration_steps has been reduced to the batch size value ({BATCH_SIZE})')
    N_integration_steps = BATCH_SIZE
if N_integration_steps<0:
    print(f'N_integration_steps < 0, N_integration_steps = BATCH_SIZE ({BATCH_SIZE})')
    N_integration_steps = BATCH_SIZE
if BATCH_SIZE%N_integration_steps!=0:
    raise Exception(f'N_integration_steps is not a divider of BATCH_SIZE: {N_integration_steps}%{BATCH_SIZE}={BATCH_SIZE%N_integration_steps}, try again')

##########################################
# EXPERIMENT 1:
#
# slab model, estimate of 
#   K0 and r
##########################################
model_name = 'slab'

# Initialization
myCoriolis = Coriolis_term(fc = fc)
myStress = Stress_term(K0 = K0, to_train=True)
myDissipation = Dissipation_Rayleigh(R = R, to_train=True)
myRHS = RHS(myCoriolis, myStress, myDissipation)

# make test and train datasets
train_data, test_data = data_maker(ds=ds,
                                    test_ratio=test_ratio, 
                                    features_names=[],
                                    forcing_names=[],
                                    dx=dx*distance_1deg_equator,
                                    dy=dy*distance_1deg_equator,
                                    mydtype=mydtype)

train_iterator = batch_loader(data_set=train_data,
                            batch_size=BATCH_SIZE)


# add path to save figures according to model name
path_save = name_base_folder+model_name+'/'
os.system(f'mkdir -p {path_save}')

if TRAIN:
    print('* Training the slab model ...')
    # optimizer = optax.adam(1e-3)
    optimizer = optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch(
                                                max_linesearch_steps=55,
                                                verbose=True))
    """optional: learning rate scheduler initialization"""

    # train loop
    lastmodel, bestmodel, Train_loss, Test_loss, opt_state_save = train(
                                                                the_model   = myRHS,
                                                                optim       = optimizer,
                                                                iter_train_data = train_iterator,
                                                                test_data   = test_data,
                                                                maxstep     = MAX_STEP,
                                                                print_every = 1,
                                                                tol         = 1e-6,
                                                                N_integration_steps = N_integration_steps,
                                                                dt          = dt_Euler,
                                                                dt_forcing  = dt_forcing,
                                                                L_to_be_normalized = [])
    if SAVE:
        eqx.tree_serialise_leaves(path_save+f'best_{model_name}.pt', bestmodel)
        
    fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
    epochs = np.arange(len(Train_loss))
    epochs_test = np.arange(len(Test_loss))*1
    ax.plot(epochs, Train_loss, label='train')
    ax.plot(epochs_test, Test_loss, label='test')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    if SAVE:
        fig.savefig(path_save+'Loss.png')
    
    print(f'Final K0 = {bestmodel.stress_term.K0}')
    print(f'Final r = {bestmodel.dissipation_term.R}')
    print('done!')

if False:
    # Plotting the solution
    """"""
    
    
    print(f'K0 = {best_RHS.stress_term.K0}')
    print(f'r = {best_RHS.dissipation_term.R}')
                                                
    dynamic_model, static_model = my_partition(best_RHS)
    n_test_data, norms = normalize_batch(test_data, 
                                  L_to_be_normalized=[]) # <- here no need to normalize as there is no NN
    
    # compute trajectories
    Nt = len(n_test_data['forcing'])
    print(Nt)
    Ntraj = Nt//(N_integration_steps_verif) -1
    L_trajectories = []
    for itraj in range(Ntraj):
        start = itraj * N_integration_steps_verif
        end = (itraj+1)*N_integration_steps_verif
        end = end if end<Nt else Nt-1

        X0 = n_test_data['forcing'][start,0:2,:,:]
        # print(X0.shape)
        my_forcing_for_this_traj = n_test_data['forcing'][start:end+1,2:4,:,:]
        my_features_for_this_traj = n_test_data['features'][start:end+1,:,:,:]
        mytraj = Integration_Euler(X0, 
                                   my_forcing_for_this_traj, 
                                   my_features_for_this_traj,
                                   best_RHS, 
                                   dt_Euler, 
                                   dt_forcing,
                                   N_integration_steps_verif,
                                   norms)
        L_trajectories.append(mytraj)
  
    xtime = np.arange(Ntrain+1, len(ds.time), 1.)*onehour/oneday
    fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
    ax.plot(xtime, n_test_data['target'].mean(axis=(2,3))[:,0], c='k', label='true U')
    for itraj in range(Ntraj):
        start = itraj * N_integration_steps_verif
        end = (itraj+1)*N_integration_steps_verif
        end = end if end<Nt else Nt-1
        # print(start, end)
        one_window = np.arange(start,end+1)*dt_forcing/oneday
        
        t0 = (Ntrain)*dt_forcing/oneday
        # print( one_window.shape, L_trajectories[itraj].mean(axis=(2,3))[:,0].shape)
        # pprint.pprint(L_trajectories[itraj].mean(axis=(2,3))[:,0])
        ax.plot(t0+ one_window, L_trajectories[itraj].mean(axis=(2,3))[:,0], c='b')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('zonal current (m/s)')
    ax.set_title('test_data')
    ax.legend()
    
    
    # integrating 1 traj
    X0 = n_test_data['forcing'][0,0:2,:,:]
    my_forcing_for_this_traj = n_test_data['forcing'][:,2:4,:,:]
    my_features_for_this_traj = n_test_data['features'][:,:,:,:]
    mytraj = Integration_Euler(X0, 
                                   my_forcing_for_this_traj, 
                                   my_features_for_this_traj,
                                   best_RHS, 
                                   dt_Euler, 
                                   dt_forcing,
                                   Nt,
                                   norms)
    xtime = np.arange(Ntrain+1, len(ds.time), 1.)*onehour/oneday
    fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
    ax.plot(xtime, n_test_data['target'].mean(axis=(2,3))[:,0], c='k', label='true U')
    ax.plot(xtime, mytraj.mean(axis=(2,3))[:,0], c='b', label='estimated U')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('zonal current (m/s)')
    ax.set_title('test_data')
    ax.legend()
    
    
if True:
    print('* map of cost function for full train dataset')
    # map of the cost function
    n_train_data, norms = normalize_batch(train_data, 
                                  L_to_be_normalized=[])
    
    R_range     = np.arange(-14, -3, 1)
    K0_range    = np.arange(-14, -3, 1)
    
    L = np.zeros((len(R_range),len(K0_range)))
    for j,r in enumerate(R_range):
        for i,k in enumerate(K0_range):
            mymodel = eqx.tree_at( lambda t:t.dissipation_term.R, myRHS, r)
            mymodel = eqx.tree_at( lambda t:t.stress_term.K0, mymodel, k)
            dynamic_model, static_model = my_partition(mymodel)
            L[j,i] = vmap_loss(dynamic_model, static_model, n_train_data, N_integration_steps, dt_Euler, dt_forcing, norms)
    
    fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
    s = ax.pcolormesh(K0_range, R_range, L, cmap='jet', norm='log',vmin=1e-3,vmax=10.)
    plt.colorbar(s, ax=ax)
    ax.set_xlabel('log(K0)')
    ax.set_ylabel('log(R)')
    ax.set_title('cost function map (train_data)')
    
if True:
    print('* testing my integration function')
    # testing the integration with my function
    #
    # soluion is app. R = -10.5, K0=-11.0
    
    mymodel = eqx.tree_at( lambda t:t.dissipation_term.R, myRHS, -10.5)
    mymodel = eqx.tree_at( lambda t:t.stress_term.K0, mymodel, -11.0)
    
    trained_model = eqx.tree_deserialise_leaves(path_save+'best_slab.pt',   # <- getting the saved PyTree 
                                                myRHS                    # <- here the call is just to get the structure
                                                )
    
    n_train_data, norms = normalize_batch(train_data, 
                                  L_to_be_normalized=[])
    
    X0 = n_train_data['forcing'][0,0:2,:,:]
    myforcing = n_train_data['forcing'][:,2:4,:,:]
    traj = Integration_Euler(X0, 
                             myforcing,
                             n_train_data['features'],
                             mymodel,
                             dt_Euler,
                             dt_forcing,
                             Ntrain,
                             norms)
    
    traj_2 = Integration_Euler(X0, 
                             myforcing,
                             n_train_data['features'],
                             trained_model,
                             dt_Euler,
                             dt_forcing,
                             Ntrain,
                             norms)
    
    xtime = np.arange(0, Ntrain, 1)*dt_forcing/oneday
    fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
    ax.plot(xtime, traj[:,0].mean(axis=(1,2)), c='b', label='slab U (by hand)')
    ax.plot(xtime, traj_2[:,0].mean(axis=(1,2)), c='g', label='slab U (by train)')
    ax.plot(xtime, n_train_data['target'][:,0].mean(axis=(1,2)), c='k', label='true U')
    ax.plot()
    ax.set_xlabel('time (days)')
    ax.set_ylabel('zonal current (m/s)')
    ax.set_title('slab, on train_data')
    ax.legend()
    
    
    
    
    
##########################################
# EXPERIMENT 2:
#
# slab model, with NN as dissipation 
#   K0 is fixed and we find theta
##########################################


# NN initialization
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)

   
if TRAIN:
    """
    """
    
#############
# PLOTTING
#############
    
if PLOTTING:
    
    """First: get back the best model(s)"""
    
    """Plot: trajectory reconstructed on train dataset (and next on test data set)
    
        -> solution K0 and r
        -> solution K0 and MLP_linear(U,V)
        -> solution K0 and non linear NN (CNN ?), using only  U and V as features
    """
    
    """Plot: comparison of the dissipation term with different parametrizations, same models as above
    """
    
    """Plot: U budget for each parametrization
    """
    
    """Plot: accuracy of the reconstruction with respect to the length of integration time, from offline to full online
    """
    
    
plt.show()

