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
from models_definition import RHS, Dissipation_Rayleigh, Stress_term, Coriolis_term, Dissipation_CNN, Dissipation_MLP, get_number_of_param
from configs import prepare_config

#====================================================================================================
# PARAMETERS
#====================================================================================================

# training the models
TRAIN_SLAB      = True     # run the training and save best model
TRAIN_NN        = True
TRAINING_MODE   = 'online'
NN_MODEL_NAME   = 'MLP'     # CNN MLP
PLOT_THE_MODEL  = True      # plot a trajectory with model converged

# Comparing models
PLOTTING        = True      # plot figures that compares models
COMPARE_TO_NN   = 'MLP'

# time steps
dt_forcing  = 3600.         # time step of forcing (s)
dt_Euler    = 60.           # time step of integration (s)

# Defining data
TEST_RATIO  = 20            # % of hours for test (over the data)
mydtype     = 'float'       # type of data, float or float32
Nsize       = 128           # size of the domain, in nx ny
dx          = 0.1           # X data resolution in ° 
dy          = 0.1           # Y data resolution in ° 

# initial guess for slab
K0          = -8.           # initial K0
R           = -8.           # initial R

# for NN reproductibility
SEED            = 5678        

path_save_png = f'./pngs/{TRAINING_MODE}/' 
#====================================================================================================

#====================================================================================================

# create folder for each set of features
name_base_folder = 'features'+''.join('_'+variable for variable in [])+'/'
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
Ntests = TEST_RATIO*len(ds.time)//100 # how many instants used for test
Ntrain = len(ds.time) - Ntests

print('* ==================================================')
print(' Global parameters:')
print(f' - TRAIN_SLAB       = {TRAIN_SLAB}')
print(f' - TRAIN_NN         = {TRAIN_NN}')
print(f' - TRAINING_MODE    = {TRAINING_MODE}')
print(f' - NN_MODEL_NAME    = {NN_MODEL_NAME}')
print(f' - PLOT_THE_MODEL   = {PLOT_THE_MODEL}')
print('* ==================================================\n')
##########################################
# EXPERIMENT 1:
#
# slab model, estimate of 
#   K0 and r
##########################################
model_name = 'slab'

my_config, my_train_config = prepare_config(ds, model_name, TRAINING_MODE, TEST_RATIO)
OPTI, MAX_STEP, PRINT_EVERY, FEATURES_NAMES, FORCING_NAMES, BATCH_SIZE, L_TO_BE_NORMALIZED = my_config
N_integration_steps, N_integration_steps_verif = my_train_config


# Initialization
myCoriolis = Coriolis_term(fc = fc)
myStress = Stress_term(K0 = K0, to_train=True)
myDissipation = Dissipation_Rayleigh(R = R, to_train=True)
myRHS = RHS(myCoriolis, myStress, myDissipation)

# make test and train datasets
train_data, test_data = data_maker(ds=ds,
                                    test_ratio=TEST_RATIO, 
                                    features_names=FEATURES_NAMES,
                                    forcing_names=FORCING_NAMES,
                                    dx=dx*distance_1deg_equator,
                                    dy=dy*distance_1deg_equator,
                                    mydtype=mydtype)
train_iterator = batch_loader(data_set=train_data,
                            batch_size=BATCH_SIZE)


# add path to save figures according to model name
path_save = name_base_folder+model_name+'/'+TRAINING_MODE+'/'
os.system(f'mkdir -p {path_save}')

if TRAIN_SLAB:
    print(f'* Training the {model_name} model ...')
    print(f'    number of param = {get_number_of_param(myRHS)}')

    # train loop
    lastmodel, bestmodel, Train_loss, Test_loss, opt_state_save = train(
                                                                the_model   = myRHS,
                                                                optim       = OPTI,
                                                                iter_train_data = train_iterator,
                                                                test_data   = test_data,
                                                                maxstep     = MAX_STEP,
                                                                print_every = PRINT_EVERY,
                                                                tol         = 1e-6,
                                                                N_integration_steps = N_integration_steps,
                                                                dt          = dt_Euler,
                                                                dt_forcing  = dt_forcing,
                                                                L_to_be_normalized = L_TO_BE_NORMALIZED)
    # save the model
    eqx.tree_serialise_leaves(path_save+f'best_RHS_{model_name}.pt', bestmodel)
        
    fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
    epochs = np.arange(len(Train_loss))
    epochs_test = np.arange(len(Test_loss))*1
    ax.plot(epochs, Train_loss, label='train')
    ax.plot(epochs_test, Test_loss, label='test')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(path_save+'Loss.png')
    
    print(f'Final K0 = {bestmodel.stress_term.K0}')
    print(f'Final r = {bestmodel.dissipation_term.R}')
    print('done!')

# Plotting the solution
if PLOT_THE_MODEL:
    print(f'* Plotting a trajectory using {model_name}')
    best_RHS = eqx.tree_deserialise_leaves(path_save+'best_RHS_slab.pt',   # <- getting the saved PyTree 
                                                myRHS                    # <- here the call is just to get the structure
                                                )
    
    print(f'K0 = {best_RHS.stress_term.K0}')
    print(f'r = {best_RHS.dissipation_term.R}')
                                                
    dynamic_model, static_model = my_partition(best_RHS)
    n_test_data, norms_te = normalize_batch(test_data, 
                                  L_to_be_normalized='') # <- here no need to normalize as there is no NN
    n_train_data, norms_tr = normalize_batch(train_data, 
                                  L_to_be_normalized='') # <- here no need to normalize as there is no NN
    
    # integrating 1 traj
    for name_data, data, norms in zip(['train','test'],[n_train_data, n_test_data],[norms_tr, norms_te]):
        Nt = len(data['forcing'])
        
        X0 = data['forcing'][0,0:2,:,:]
        my_forcing_for_this_traj = data['forcing'][:,2:4,:,:]
        my_features_for_this_traj = data['features'][:,:,:,:]
        mytraj = Integration_Euler(X0, 
                                    my_forcing_for_this_traj, 
                                    my_features_for_this_traj,
                                    best_RHS, 
                                    dt_Euler, 
                                    dt_forcing,
                                    Nt,
                                    norms)
        if name_data=='test':
            xtime = np.arange(Ntrain, len(ds.time), 1.)*onehour/oneday
        elif name_data=='train':
            xtime = np.arange(0, Ntrain, 1.)*onehour/oneday
        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
        ax.plot(xtime, data['target'].mean(axis=(2,3))[:,0], c='k', label='true U')
        ax.plot(xtime, mytraj.mean(axis=(2,3))[:,0], c='b', label='estimated U')
        ax.set_xlabel('time (days)')
        ax.set_ylabel('zonal current (m/s)')
        ax.set_title(f'{model_name}: {name_data}_data')
        ax.legend()
        fig.savefig(path_save+f'traj_{name_data}.png')
   
# map of the cost function
if False:
    print('* map of cost function for full train dataset')
   
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
    fig.savefig(path_save+'cost_function_map.png')
    

##########################################
# EXPERIMENT 2:
#
# slab model, with NN as dissipation 
#   K0 is fixed and we find theta
##########################################
model_name = NN_MODEL_NAME

my_config, my_train_config = prepare_config(ds, model_name, TRAINING_MODE, TEST_RATIO)
OPTI, MAX_STEP, PRINT_EVERY, FEATURES_NAMES, FORCING_NAMES, BATCH_SIZE, L_TO_BE_NORMALIZED = my_config
N_integration_steps, N_integration_steps_verif = my_train_config

# Initialization
# replace the K0 value with the one from the slab (experiment 1 above)
my_RHS_slab = eqx.tree_deserialise_leaves(name_base_folder+'slab/'+TRAINING_MODE+'/best_RHS_slab.pt',
                                                 RHS(Coriolis_term(fc = fc), 
                                                     Stress_term(K0 = K0), 
                                                     Dissipation_Rayleigh(R = R))                  
                                                )
K0_slab = my_RHS_slab.stress_term.K0
myCoriolis = Coriolis_term(fc = fc)
myStress = Stress_term(K0 = K0_slab, to_train=True)

key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
if NN_MODEL_NAME=='CNN':
    myDissipation = Dissipation_CNN(subkey, len(FEATURES_NAMES)+2, to_train=True)
elif NN_MODEL_NAME=='MLP':
    myDissipation = Dissipation_MLP(subkey, len(FEATURES_NAMES)+2, to_train=True)
else:
    raise Exception(f'NN_MODEL_NAME={NN_MODEL_NAME} is not recognized')

myRHS = RHS(myCoriolis, myStress, myDissipation)
# make test and train datasets
train_data, test_data = data_maker(ds=ds,
                                    test_ratio=TEST_RATIO, 
                                    features_names=FEATURES_NAMES,
                                    forcing_names=FORCING_NAMES,
                                    dx=dx*distance_1deg_equator,
                                    dy=dy*distance_1deg_equator,
                                    mydtype=mydtype)
train_iterator = batch_loader(data_set=train_data,
                            batch_size=BATCH_SIZE)


# add path to save figures according to model name
path_save = name_base_folder+model_name+'/'+TRAINING_MODE+'/'
os.system(f'mkdir -p {path_save}')

   
if TRAIN_NN:
    print(f'* Training the {model_name} model ...')
    print(f'    number of param = {get_number_of_param(myRHS)}')
    # train loop
    lastmodel, bestmodel, Train_loss, Test_loss, opt_state_save = train(
                                                                the_model   = myRHS,
                                                                optim       = OPTI,
                                                                iter_train_data = train_iterator,
                                                                test_data   = test_data,
                                                                maxstep     = MAX_STEP,
                                                                print_every = PRINT_EVERY,
                                                                retol       = 0.0001,
                                                                N_integration_steps = N_integration_steps,
                                                                dt          = dt_Euler,
                                                                dt_forcing  = dt_forcing,
                                                                L_to_be_normalized = L_TO_BE_NORMALIZED)
    # save the model
    eqx.tree_serialise_leaves(path_save+f'best_RHS_{model_name}.pt', bestmodel)
        
    fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
    epochs = np.arange(len(Train_loss))
    epochs_test = np.arange(len(Test_loss))*1
    ax.plot(epochs, Train_loss, label='train')
    ax.plot(epochs_test, Test_loss, label='test')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(path_save+'Loss.png')
    print('done!')

# plots a trajectory with the best model 
if PLOT_THE_MODEL:
    print(f'* Plotting a trajectory using {model_name}')
    
    if NN_MODEL_NAME=='CNN':
        myDissipation = Dissipation_CNN(subkey, len(FEATURES_NAMES)+2, to_train=True)
    elif NN_MODEL_NAME=='MLP':
        myDissipation = Dissipation_MLP(subkey, len(FEATURES_NAMES)+2, to_train=True)
    else:
        raise Exception(f'NN_MODEL_NAME={NN_MODEL_NAME} is not recognized')
    best_RHS = eqx.tree_deserialise_leaves(name_base_folder+COMPARE_TO_NN+'/'+TRAINING_MODE+f'/best_RHS_{COMPARE_TO_NN}.pt',
                                                 RHS(Coriolis_term(fc = fc), 
                                                    Stress_term(K0 = K0_slab), 
                                                     myDissipation)                  
                                                )

    dynamic_model, static_model = my_partition(best_RHS)
    n_test_data, norms_te = normalize_batch(test_data, 
                                  L_to_be_normalized='features')
    n_train_data, norms_tr = normalize_batch(train_data, 
                                  L_to_be_normalized='features')
    
    # integrating 1 traj
    for name_data, data, norms in zip(['train','test'],[n_train_data, n_test_data],[norms_tr, norms_te]):
        Nt = len(data['forcing'])
        
        X0 = data['forcing'][0,0:2,:,:]
        my_forcing_for_this_traj = data['forcing'][:,2:4,:,:]
        my_features_for_this_traj = data['features'][:,:,:,:]
        mytraj = Integration_Euler(X0, 
                                    my_forcing_for_this_traj, 
                                    my_features_for_this_traj,
                                    best_RHS, 
                                    dt_Euler, 
                                    dt_forcing,
                                    Nt,
                                    norms)
        if name_data=='test':
            xtime = np.arange(Ntrain, len(ds.time), 1.)*onehour/oneday
        elif name_data=='train':
            xtime = np.arange(0, Ntrain, 1.)*onehour/oneday
        fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=100)
        ax.plot(xtime, data['target'].mean(axis=(2,3))[:,0], c='k', label='true U')
        ax.plot(xtime, mytraj.mean(axis=(2,3))[:,0], c='b', label='estimated U')
        ax.set_xlabel('time (days)')
        ax.set_ylabel('zonal current (m/s)')
        ax.set_title(f'{model_name}: {name_data}_data')
        ax.set_ylim([-0.2,0.2])
        ax.legend()
        fig.savefig(path_save+f'traj_{name_data}.png')
    
    
#############
# PLOTTING
#############

if PLOTTING:
    os.system(f'mkdir -p {path_save_png}')
    colors = ['b','g']
    
    # NN output
    if NN_MODEL_NAME=='CNN':
        myDissipation = Dissipation_CNN(subkey, len(FEATURES_NAMES)+2, to_train=True)
    elif NN_MODEL_NAME=='MLP':
        myDissipation = Dissipation_MLP(subkey, len(FEATURES_NAMES)+2, to_train=True)
    else:
        raise Exception(f'NN_MODEL_NAME={NN_MODEL_NAME} is not recognized')
    best_RHS_NN = eqx.tree_deserialise_leaves(name_base_folder+COMPARE_TO_NN+'/'+TRAINING_MODE+f'/best_RHS_{COMPARE_TO_NN}.pt',
                                                 RHS(Coriolis_term(fc = fc), 
                                                     Stress_term(K0 = K0), 
                                                     myDissipation)                  
                                                )
    dynamic_model, static_model = my_partition(best_RHS_NN)
    NN_test_data, NN_norms_te = normalize_batch(test_data, 
                                  L_to_be_normalized='features')
    NN_train_data, NN_norms_tr = normalize_batch(train_data, 
                                  L_to_be_normalized='features')
    
    traj_NN_train = Integration_Euler(NN_train_data['forcing'][0,0:2,:,:], 
                                    NN_train_data['forcing'][:,2:4,:,:], 
                                    NN_train_data['features'],
                                    best_RHS_NN, 
                                    dt_Euler, 
                                    dt_forcing,
                                    NN_train_data['forcing'].shape[0],
                                    NN_norms_tr)
    traj_NN_test = Integration_Euler(NN_test_data['forcing'][0,0:2,:,:], 
                                    NN_test_data['forcing'][:,2:4,:,:], 
                                    NN_test_data['features'],
                                    best_RHS_NN, 
                                    dt_Euler, 
                                    dt_forcing,
                                    NN_test_data['forcing'].shape[0],
                                    NN_norms_te)
    
    
    # slab output
    best_RHS_slab =  eqx.tree_deserialise_leaves(name_base_folder+'slab/'+TRAINING_MODE+'/best_RHS_slab.pt',   # <- getting the saved PyTree 
                                                RHS(Coriolis_term(fc = fc), 
                                                    Stress_term(K0 = K0), 
                                                    Dissipation_Rayleigh(R = R))                    # <- here the call is just to get the structure
                                                ) 
    
    
                                                   
    dynamic_model, static_model = my_partition(best_RHS_slab)
    slab_test_data, slab_norms_te = normalize_batch(test_data, 
                                  L_to_be_normalized='') # <- here no need to normalize as there is no NN
    slab_train_data, slab_norms_tr = normalize_batch(train_data, 
                                  L_to_be_normalized='') # <- here no need to normalize as there is no NN
    
    traj_slab_train = Integration_Euler(slab_train_data['forcing'][0,0:2,:,:], 
                                    slab_train_data['forcing'][:,2:4,:,:], 
                                    slab_train_data['features'],
                                    best_RHS_slab, 
                                    dt_Euler, 
                                    dt_forcing,
                                    slab_train_data['forcing'].shape[0],
                                    slab_norms_tr)
    traj_slab_test = Integration_Euler(slab_test_data['forcing'][0,0:2,:,:], 
                                    slab_test_data['forcing'][:,2:4,:,:], 
                                    slab_test_data['features'],
                                    best_RHS_slab, 
                                    dt_Euler, 
                                    dt_forcing,
                                    slab_test_data['forcing'].shape[0],
                                    slab_norms_te)
    
    
    # train set
    xtime = np.arange(0, Ntrain, 1.)*onehour/oneday
    fig, ax = plt.subplots(2,1,figsize = (10,10), constrained_layout=True,dpi=100)
    ax[0].plot(xtime, slab_train_data['target'].mean(axis=(2,3))[:,0], c='k', label='true U')
    ax[0].plot(xtime, traj_slab_train.mean(axis=(2,3))[:,0], c='b', label='estimated U (slab)')
    ax[0].plot(xtime, traj_NN_train.mean(axis=(2,3))[:,0], c='g', label='estimated U (NN)')
    ax[0].set_ylabel('zonal current (m/s)')
    ax[0].set_title(f'train_data')
    ax[0].set_ylim([-0.2,0.2])
    ax[0].legend()
    ax[1].plot(xtime, slab_train_data['forcing'].mean(axis=(2,3))[:,2], c='b', label='TAx')
    ax[1].plot(xtime, slab_train_data['forcing'].mean(axis=(2,3))[:,3], c='orange', label='TAy')
    ax[1].set_xlabel('time (days)')
    ax[1].set_ylabel('stress N/m2')
    ax[1].legend()
    ax[1].set_ylim([-2.,2.])
    fig.savefig(path_save_png+'train_set.png')
    
    # test set
    xtime = np.arange(Ntrain, len(ds.time), 1.)*onehour/oneday
    fig, ax = plt.subplots(2,1,figsize = (10,10), constrained_layout=True,dpi=100)
    ax[0].plot(xtime, slab_test_data['target'].mean(axis=(2,3))[:,0], c='k', label='true U')
    ax[0].plot(xtime, traj_slab_test.mean(axis=(2,3))[:,0], c='b', label='estimated U (slab)')
    ax[0].plot(xtime, traj_NN_test.mean(axis=(2,3))[:,0], c='g', label='estimated U (NN)')
    ax[0].set_ylabel('zonal current (m/s)')
    ax[0].set_title(f'test_data')
    ax[0].set_ylim([-0.2,0.2])
    ax[0].legend()
    ax[1].plot(xtime, slab_test_data['forcing'].mean(axis=(2,3))[:,2], c='b', label='TAx')
    ax[1].plot(xtime, slab_test_data['forcing'].mean(axis=(2,3))[:,3], c='orange', label='TAy')
    ax[1].set_xlabel('time (days)')
    ax[1].set_ylabel('stress N/m2')
    ax[1].set_ylim([-2.,2.])
    ax[1].legend()
    ax[1].legend()
    fig.savefig(path_save_png+'test_set.png')
  
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

