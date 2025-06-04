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
    
    
to be explored : better conditionning of the problem
  https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c8ac317ce6b43ecebf4ff67c1529ff8ef4cf1a86


-> les questions en cours 04/06/25 Hugo:
- faut il normaliser les inputs, denormaliser les ouputs d'un NN ?
- comment avoir un entrainement des poids de la même maniere pour tes les poids ?
- comment mixer l'entrainement d'un NN et l'estimation d'un parametre dans une equation physique ?
    example de K0 et les poids d'un NN pour le slab

"""

# regular modules import
from re import L
import xarray as xr
import numpy as np
import jax
import equinox as eqx
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '../../src')
# jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)
from jax import lax
from scipy.stats import gaussian_kde

# my imports
from train_preprocessing import data_maker, batch_loader, normalize_batch, get_dataset_subset
from train_functions import train, my_partition, vmap_loss
from time_integration import Integration_Euler
from constants import oneday, onehour, distance_1deg_equator
from models_definition import *
from configs import prepare_config

#====================================================================================================
# PARAMETERS
#====================================================================================================

# => see configs.py for configuration specific parameters

# training the models
TRAIN_SLAB      = False     # run the training and save best model
TRAIN_NN        = True
TRAINING_MODE   = 'online'
NN_MODEL_NAME   = 'MLP_linear'     # CNN MLP MLP_linear
USE_AMPLITUDE   = False      # loss on amplitude of currents (True) or on currents themselves (False)
PLOT_THE_MODEL  = False      # plot a trajectory with model converged

# Comparing models
PLOTTING        = True      # plot figures that compares models
COMPARE_TO_NN   = NN_MODEL_NAME

# time steps
dt_forcing  = 3600.         # time step of forcing (s)
dt_Euler    = 60.           # time step of integration (s)

# Defining data
TEST_RATIO  = 20            # % of hours for test (over the data)
mydtype     = 'float'       # type of data, float or float32
Nsize       = 1           # size of the domain, in nx ny
Nsize_extra = 64
dx          = 0.1           # X data resolution in ° 
dy          = 0.1           # Y data resolution in ° 

# initial guess for slab
K0          = -8.           # initial K0
R           = -8.           # initial R

# for NN reproductibility
SEED            = 5678        

path_save_png = f'./pngs/{TRAINING_MODE}/' 

"""
TO DO:

- Link physical term / NN term:
    Last linear layer of NN term is initialized with a normal shape, with the std being very tricky to set.

- With the first NN architectures, results are worse than slab model -> tune that ? hyperparameters ?

- add the unsteak 2 layer for comparison ?

- add RMSE value for each model in the comparison
"""



#====================================================================================================

#====================================================================================================

# create folder for each set of features
if USE_AMPLITUDE:
    add_txt_amplitude = 'on_amplitude/'
else:
    add_txt_amplitude = 'on_currents/'
name_base_folder = add_txt_amplitude +'features'+''.join('_'+variable for variable in [])+'/'
os.system(f'mkdir -p {name_base_folder}')

filename =  [
            # '../../data_regrid/croco_1h_inst_surf_2005-01-01-2005-01-31_0.1deg_conservative.nc',
            #   '../../data_regrid/croco_1h_inst_surf_2005-02-01-2005-02-28_0.1deg_conservative.nc',
              '../../data_regrid/croco_1h_inst_surf_2005-03-01-2005-03-31_0.1deg_conservative.nc',
            #   '../../data_regrid/croco_1h_inst_surf_2005-04-01-2005-04-30_0.1deg_conservative.nc',
            #   '../../data_regrid/croco_1h_inst_surf_2005-05-01-2005-05-31_0.1deg_conservative.nc',
            #   '../../data_regrid/croco_1h_inst_surf_2005-06-01-2005-06-30_0.1deg_conservative.nc',
            #   '../../data_regrid/croco_1h_inst_surf_2005-07-01-2005-07-31_0.1deg_conservative.nc',
            #   '../../data_regrid/croco_1h_inst_surf_2005-08-01-2005-08-31_0.1deg_conservative.nc',
            #   '../../data_regrid/croco_1h_inst_surf_2005-09-01-2005-09-30_0.1deg_conservative.nc',
            #   '../../data_regrid/croco_1h_inst_surf_2005-10-01-2005-10-31_0.1deg_conservative.nc',
            #   '../../data_regrid/croco_1h_inst_surf_2005-11-01-2005-11-30_0.1deg_conservative.nc',
            #   '../../data_regrid/croco_1h_inst_surf_2005-12-01-2005-12-31_0.1deg_conservative.nc'
              ]

# prepare data
ds = xr.open_mfdataset(filename)
ds = ds.isel(lon=slice(-(Nsize+Nsize_extra),-Nsize_extra),lat=slice(Nsize_extra,(Nsize+Nsize_extra))) # <- this could be centered on a physical location
ds = ds.rename({'oceTAUX':'TAx', 'oceTAUY':'TAy'})
fc = np.asarray(2*2*np.pi/86164*np.sin(ds.lat.values*np.pi/180))
Ntests = TEST_RATIO*len(ds.time)//100 # how many instants used for test
Ntrain = len(ds.time) - Ntests

print('* ==================================================')
print(' Global parameters:')
print(f' - TRAIN_SLAB       = {TRAIN_SLAB}')
print(f' - TRAIN_NN         = {TRAIN_NN}')
print(f' - TRAINING_MODE    = {TRAINING_MODE}')
print(f' - USE_AMPLITUDE    = {USE_AMPLITUDE}')
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
                                                                tol         = 1e-5,
                                                                retol       = 1e-4,
                                                                N_integration_steps = N_integration_steps,
                                                                dt          = dt_Euler,
                                                                dt_forcing  = dt_forcing,
                                                                L_to_be_normalized = L_TO_BE_NORMALIZED,
                                                                use_amplitude = USE_AMPLITUDE)
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
   
    n_train_data, norms = normalize_batch(next(train_iterator),  # train_data
                                  L_to_be_normalized='')
    
    R_range     = np.arange(-20, -3, 1)
    K0_range    = np.arange(-20, -3, 1)
    
    def fn_for_vmap(mymodel, K0, r):
            mymodel = eqx.tree_at( lambda t:t.dissipation_term.R, mymodel, r)
            mymodel = eqx.tree_at( lambda t:t.stress_term.K0, mymodel, K0)
            dyn,stat = my_partition(mymodel)
            return vmap_loss(dyn, stat, n_train_data, N_integration_steps, dt_Euler, dt_forcing, norms, USE_AMPLITUDE)
        
    J = jax.vmap(jax.vmap(fn_for_vmap, in_axes=(None, None, 0)),
                                        in_axes=(None, 0, None))(myRHS, K0_range, R_range)
    
    
    
    fig, ax = plt.subplots(1,1,figsize = (5,5), constrained_layout=True,dpi=100)
    s = ax.pcolormesh(K0_range, R_range, J, cmap='terrain', norm='log',vmin=1e-1,vmax=10.)
    plt.colorbar(s, ax=ax)
    ax.set_xlabel('log(K0)')
    ax.set_ylabel('log(R)')
    ax.set_aspect(1)
    ax.set_title('cost function map (train_data)')
    fig.savefig(path_save+'cost_function_map.png')
    plt.show()
    raise Exception

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
my_RHS_slab = eqx.tree_deserialise_leaves(path_save+f'best_RHS_slab.pt',
                                                 RHS(Coriolis_term(fc = fc), 
                                                     Stress_term(K0 = K0), 
                                                     Dissipation_Rayleigh(R = R))                  
                                                )
K0_slab = my_RHS_slab.stress_term.K0
myCoriolis = Coriolis_term(fc = fc)
myStress = Stress_term(K0 = K0_slab, to_train=True) # False

key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
if NN_MODEL_NAME=='CNN':
    myDissipation = Dissipation_CNN(subkey, len(FEATURES_NAMES)+2, to_train=True)
elif NN_MODEL_NAME=='MLP':
    myDissipation = Dissipation_MLP(subkey, len(FEATURES_NAMES)+2, to_train=True)
elif NN_MODEL_NAME=='MLP_linear':
    # myDissipation = Dissipation_MLP_linear(subkey, len(FEATURES_NAMES)+2, to_train=True)
    myDissipation = Dissipation_MLP_linear_1D(subkey, len(FEATURES_NAMES)+2, to_train=True)
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
    print('K0 value is',myRHS.stress_term.K0)
    # train loop
    lastmodel, bestmodel, Train_loss, Test_loss, opt_state_save = train(
                                                                the_model   = myRHS,
                                                                optim       = OPTI,
                                                                iter_train_data = train_iterator,
                                                                test_data   = test_data,
                                                                maxstep     = MAX_STEP,
                                                                print_every = PRINT_EVERY,
                                                                tol         = 1e-5,
                                                                retol       = 1e-4, # 0.0001
                                                                N_integration_steps = N_integration_steps,
                                                                dt          = dt_Euler,
                                                                dt_forcing  = dt_forcing,
                                                                L_to_be_normalized = L_TO_BE_NORMALIZED,
                                                                use_amplitude = USE_AMPLITUDE)
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
    elif NN_MODEL_NAME=='MLP_linear':
        myDissipation = Dissipation_MLP_linear(subkey, len(FEATURES_NAMES)+2, to_train=True)
    else:
        raise Exception(f'NN_MODEL_NAME={NN_MODEL_NAME} is not recognized')
    best_RHS = eqx.tree_deserialise_leaves(name_base_folder+COMPARE_TO_NN+'/'+TRAINING_MODE+f'/best_RHS_{COMPARE_TO_NN}.pt',
                                                 RHS(Coriolis_term(fc = fc), 
                                                    Stress_term(K0 = K0_slab), 
                                                     myDissipation)                  
                                                )

    dynamic_model, static_model = my_partition(best_RHS)
    n_test_data, norms_te = normalize_batch(test_data, 
                                  L_to_be_normalized=L_TO_BE_NORMALIZED)
    n_train_data, norms_tr = normalize_batch(train_data, 
                                  L_to_be_normalized=L_TO_BE_NORMALIZED)
    
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
    print('* Plotting ....')
    # os.system(f'mkdir -p {path_save_png}')
    
    if USE_AMPLITUDE:
        path_save_png = 'on_amplitude/'+path_save_png
    else:
        path_save_png = 'on_currents/'+path_save_png
    os.system(f'mkdir -p {path_save_png}')
    
    
    
    colors = ['b','g']
    
    t0 = 0
    t1 = 596 #31*24
    t0_test = 0
    t1_test = 148 #291
    
    ###########
    # NN output
    ###########
    print('     -> computing trajectory using NN model')
    if NN_MODEL_NAME=='CNN':
        myDissipation = Dissipation_CNN(subkey, len(FEATURES_NAMES)+2, to_train=True)
    elif NN_MODEL_NAME=='MLP':
        myDissipation = Dissipation_MLP(subkey, len(FEATURES_NAMES)+2, to_train=True)
    elif NN_MODEL_NAME=='MLP_linear':
        # myDissipation = Dissipation_MLP_linear(subkey, len(FEATURES_NAMES)+2, to_train=True)
        myDissipation = Dissipation_MLP_linear_1D(subkey, len(FEATURES_NAMES)+2, to_train=True)
    else:
        raise Exception(f'NN_MODEL_NAME={NN_MODEL_NAME} is not recognized')
    best_RHS_NN = eqx.tree_deserialise_leaves(name_base_folder+COMPARE_TO_NN+'/'+TRAINING_MODE+f'/best_RHS_{COMPARE_TO_NN}.pt',
                                                 RHS(Coriolis_term(fc = fc), 
                                                     Stress_term(K0 = K0_slab), 
                                                     myDissipation)                  
                                                )
    print('K0 value is',best_RHS_NN.stress_term.K0)
    # here we get a subset as for long integration (= long rollout)
    #   the data doesnt fit in memory
    data_train_plot = get_dataset_subset(data_set=train_data,
                                        t0 = t0,
                                        t1 = t1)
    data_test_plot = get_dataset_subset(data_set=test_data,
                                        t0 = t0_test,
                                        t1 = t1_test)
        
    # NN_test_data, NN_norms_te = normalize_batch(test_data, 
    #                               L_to_be_normalized=L_TO_BE_NORMALIZED)
    # NN_train_data, NN_norms_tr = normalize_batch(train_data, 
    #                               L_to_be_normalized=L_TO_BE_NORMALIZED)
    
    NN_train_data, NN_norms_tr = normalize_batch(data_train_plot, 
                                  L_to_be_normalized=L_TO_BE_NORMALIZED)
    NN_test_data, NN_norms_te = normalize_batch(data_test_plot, 
                                  L_to_be_normalized=L_TO_BE_NORMALIZED)
    
    

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
                                    NN_norms_te) # time, current, ny, nx
    
   
    
    dCdt_NN_train = np.gradient(traj_NN_train, 3600., axis=0)
    dCdt_NN_test = np.gradient(traj_NN_test, 3600., axis=0)
    coriolis_NN_train = jax.vmap(best_RHS_NN.coriolis_term)(traj_NN_train)
    coriolis_NN_test = jax.vmap(best_RHS_NN.coriolis_term)(traj_NN_test)
    stress_NN_train = jax.vmap(best_RHS_NN.stress_term)(NN_train_data['forcing'][:,2:4,:,:])
    stress_NN_test = jax.vmap(best_RHS_NN.stress_term)(NN_test_data['forcing'][:,2:4,:,:])
    # Now we need to construct inputs for the NN
    #   -> they need to be normed
    #   -> and we want to reuse the trajectory computed just above
    #       so we normalize it then add the other features (already normed)
    #   -> need to use lax.slice_in_dim that is equivalent to array[:,2:,:,:]
    normed_traj_NN_train = (traj_NN_train - NN_norms_tr['features']['mean'][np.newaxis,0:2,np.newaxis,np.newaxis])/NN_norms_tr['features']['std'][np.newaxis,0:2,np.newaxis,np.newaxis]
    normed_features = jnp.concatenate([normed_traj_NN_train, 
                                 lax.slice_in_dim(NN_train_data['features'], 
                                                  start_index=2, 
                                                  limit_index=NN_train_data['features'].shape[1], 
                                                  axis=1)],
                                axis=1) # 
    diss_NN_train = jax.vmap(best_RHS_NN.dissipation_term)(normed_features)
    # now same for test data:
    normed_traj_NN_test = (traj_NN_test - NN_norms_te['features']['mean'][np.newaxis,0:2,np.newaxis,np.newaxis])/NN_norms_te['features']['std'][np.newaxis,0:2,np.newaxis,np.newaxis]
    normed_features = jnp.concatenate([normed_traj_NN_test, 
                                 lax.slice_in_dim(NN_test_data['features'], 
                                                  start_index=2, 
                                                  limit_index=NN_train_data['features'].shape[1], 
                                                  axis=1)],
                                axis=1)
    diss_NN_test = jax.vmap(best_RHS_NN.dissipation_term)(normed_features)
    
    #############
    # slab output
    #############
    print('     -> computing trajectory using slab model')
    best_RHS_slab =  eqx.tree_deserialise_leaves(name_base_folder+'slab/'+TRAINING_MODE+'/best_RHS_slab.pt',   # <- getting the saved PyTree 
                                                RHS(Coriolis_term(fc = fc), 
                                                    Stress_term(K0 = K0), 
                                                    Dissipation_Rayleigh(R = R))                    # <- here the call is just to get the structure
                                                ) 
    
    
    print('r value is',best_RHS_slab.dissipation_term.R,'=> exp(r) =', np.exp(best_RHS_slab.dissipation_term.R))
    # here we get a subset as for long integration (= long rollout)
    #   the data doesnt fit in memory
    data_train_plot = get_dataset_subset(data_set=train_data,
                                        t0 = t0,
                                        t1 = t1)
    data_test_plot = get_dataset_subset(data_set=test_data,
                                        t0 = t0_test,
                                        t1 = t1_test)
        
    # slab_test_data, slab_norms_te = normalize_batch(test_data, 
    #                               L_to_be_normalized='') # <- here no need to normalize as there is no NN
    # slab_train_data, slab_norms_tr = normalize_batch(train_data, 
    #                               L_to_be_normalized='') # <- here no need to normalize as there is no NN
    
    
    slab_train_data, slab_norms_tr = normalize_batch(data_train_plot, 
                                  L_to_be_normalized='') # <- here no need to normalize as there is no NN
    slab_test_data, slab_norms_te = normalize_batch(data_test_plot, 
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
    
    dCdt_slab_train = np.gradient(traj_slab_train, 3600., axis=0)
    dCdt_slab_test = np.gradient(traj_slab_test, 3600., axis=0)
    coriolis_slab_train = jax.vmap(best_RHS_slab.coriolis_term)(traj_slab_train)
    coriolis_slab_test = jax.vmap(best_RHS_slab.coriolis_term)(traj_slab_test)
    stress_slab_train = jax.vmap(best_RHS_slab.stress_term)(slab_train_data['forcing'][:,2:4,:,:])
    stress_slab_test = jax.vmap(best_RHS_slab.stress_term)(slab_test_data['forcing'][:,2:4,:,:])
    diss_slab_train = jax.vmap(best_RHS_slab.dissipation_term)(traj_slab_train)
    diss_slab_test = jax.vmap(best_RHS_slab.dissipation_term)(traj_slab_test)
    
    # true budget (at K0 fixed)
    coriolis_true_train = jax.vmap(best_RHS_slab.coriolis_term)(slab_train_data['forcing'][:,0:2,:,:])
    coriolis_true_test = jax.vmap(best_RHS_slab.coriolis_term)(slab_test_data['forcing'][:,0:2,:,:])
    stress_true_train = stress_slab_train # and = stress_NN_train
    stress_true_test = stress_slab_test # and = stress_NN_test
    dCdt_train = np.gradient(slab_train_data['forcing'][:,0:2,:,:], 3600., axis=0)
    dCdt_test = np.gradient(slab_test_data['forcing'][:,0:2,:,:], 3600., axis=0)
    diss_true_train = dCdt_train - coriolis_true_train - stress_true_train
    diss_true_test = dCdt_test - coriolis_true_test - stress_true_test
    
    if True:
        # trajectory on train set
        # xtime = np.arange(0, Ntrain, 1.)*onehour/oneday
        xtime_train = np.arange(t0,t1,1)*onehour/oneday
        fig, ax = plt.subplots(2,1,figsize = (10,10), constrained_layout=True,dpi=100)
        ax[0].plot(xtime_train, slab_train_data['target'].mean(axis=(2,3))[:,0], c='k', label='true U')
        ax[0].plot(xtime_train, traj_slab_train.mean(axis=(2,3))[:,0], c='b', label='estimated U (slab)')
        ax[0].plot(xtime_train, traj_NN_train.mean(axis=(2,3))[:,0], c='g', label=f'estimated U ({COMPARE_TO_NN})')
        ax[0].set_ylabel('zonal current (m/s)')
        ax[0].set_title(f'train_data')
        ax[0].set_ylim([-0.2,0.2])
        ax[0].legend()
        ax[1].plot(xtime_train, slab_train_data['forcing'].mean(axis=(2,3))[:,2], c='b', label='TAx')
        ax[1].plot(xtime_train, slab_train_data['forcing'].mean(axis=(2,3))[:,3], c='orange', label='TAy')
        ax[1].set_xlabel('time (days)')
        ax[1].set_ylabel('stress N/m2')
        ax[1].legend()
        ax[1].set_ylim([-2.,2.])
        fig.savefig(path_save_png+f'train_slab_{COMPARE_TO_NN}.png')
        
        # trajectory on test set
        # xtime = np.arange(Ntrain, len(ds.time), 1.)*onehour/oneday
        xtime_test = np.arange(Ntrain+t0_test,Ntrain+t1_test,1)*onehour/oneday
        fig, ax = plt.subplots(2,1,figsize = (10,10), constrained_layout=True,dpi=100)
        ax[0].plot(xtime_test, slab_test_data['target'].mean(axis=(2,3))[:,0], c='k', label='true U')
        ax[0].plot(xtime_test, traj_slab_test.mean(axis=(2,3))[:,0], c='b', label='estimated U (slab)')
        ax[0].plot(xtime_test, traj_NN_test.mean(axis=(2,3))[:,0], c='g', label=f'estimated U ({COMPARE_TO_NN})')
        ax[0].set_ylabel('zonal current (m/s)')
        ax[0].set_title(f'test_data')
        ax[0].set_ylim([-0.2,0.2])
        ax[0].legend()
        ax[1].plot(xtime_test, slab_test_data['forcing'].mean(axis=(2,3))[:,2], c='b', label='TAx')
        ax[1].plot(xtime_test, slab_test_data['forcing'].mean(axis=(2,3))[:,3], c='orange', label='TAy')
        ax[1].set_xlabel('time (days)')
        ax[1].set_ylabel('stress N/m2')
        ax[1].set_ylim([-2.,2.])
        ax[1].legend()
        ax[1].legend()
        fig.savefig(path_save_png+f'test_set_slab_{COMPARE_TO_NN}.png')
    
        # budget on train set
        fig, ax = plt.subplots(3,1,figsize = (10,10), constrained_layout=True,dpi=100)
        # -> slab
        ax[0].plot(xtime_train, coriolis_slab_train.mean(axis=(2,3))[:,0], c='orange')
        ax[0].plot(xtime_train, stress_slab_train.mean(axis=(2,3))[:,0], c='green')
        ax[0].plot(xtime_train, diss_slab_train.mean(axis=(2,3))[:,0], c='b')
        ax[0].plot(xtime_train, dCdt_slab_train.mean(axis=(2,3))[:,0], c='k')
        ax[0].set_title('slab')
        # -> NN
        ax[1].plot(xtime_train, coriolis_NN_train.mean(axis=(2,3))[:,0], c='orange')
        ax[1].plot(xtime_train, stress_NN_train.mean(axis=(2,3))[:,0], c='green')
        ax[1].plot(xtime_train, diss_NN_train.mean(axis=(2,3))[:,0], c='b')
        ax[1].plot(xtime_train, dCdt_NN_train.mean(axis=(2,3))[:,0], c='k')
        ax[1].set_title(f'NN ({COMPARE_TO_NN})')
        # -> truth
        ax[2].plot(xtime_train, coriolis_true_train.mean(axis=(2,3))[:,0], c='orange', label='Coriolis')
        ax[2].plot(xtime_train, stress_true_train.mean(axis=(2,3))[:,0], c='green', label='Stress')
        ax[2].plot(xtime_train, diss_true_train.mean(axis=(2,3))[:,0], c='b', label='Diss')
        ax[2].plot(xtime_train, dCdt_train.mean(axis=(2,3))[:,0], c='k', label='dCdt')
        ax[2].set_title('truth')
        ax[2].set_xlabel('time (days)')
        ax[2].legend()
        for axe in ax:
            axe.set_ylabel(r'$m.s^{-2}$')
            # axe.set_ylim([-2e-5,2e-4])
            axe.set_ylim([-5e-5,5e-5])
        fig.suptitle('U budget train set')
        fig.savefig(path_save_png+f'train_set_U_budget_slab_{COMPARE_TO_NN}.png')
        
        
        # budget on test set
        fig, ax = plt.subplots(3,1,figsize = (10,10), constrained_layout=True,dpi=100)
        # -> slab
        ax[0].plot(xtime_test, coriolis_slab_test.mean(axis=(2,3))[:,0], c='orange')
        ax[0].plot(xtime_test, stress_slab_test.mean(axis=(2,3))[:,0], c='green')
        ax[0].plot(xtime_test, diss_slab_test.mean(axis=(2,3))[:,0], c='b')
        ax[0].plot(xtime_test, dCdt_slab_test.mean(axis=(2,3))[:,0], c='k')
        ax[0].set_title('slab')
        # -> NN
        ax[1].plot(xtime_test, coriolis_NN_test.mean(axis=(2,3))[:,0], c='orange')
        ax[1].plot(xtime_test, stress_NN_test.mean(axis=(2,3))[:,0], c='green')
        ax[1].plot(xtime_test, diss_NN_test.mean(axis=(2,3))[:,0], c='b')
        ax[1].plot(xtime_test, dCdt_NN_test.mean(axis=(2,3))[:,0], c='k')
        ax[1].set_title(f'NN ({COMPARE_TO_NN})')
        # -> truth
        ax[2].plot(xtime_test, coriolis_true_test.mean(axis=(2,3))[:,0], c='orange', label='Coriolis')
        ax[2].plot(xtime_test, stress_true_test.mean(axis=(2,3))[:,0], c='green', label='Stress')
        ax[2].plot(xtime_test, diss_true_test.mean(axis=(2,3))[:,0], c='b', label='Diss')
        ax[2].plot(xtime_test, dCdt_test.mean(axis=(2,3))[:,0], c='k', label='dCdt')
        ax[2].set_title('truth')
        ax[2].set_xlabel('time (days)')
        ax[2].legend()
        for axe in ax:
            axe.set_ylabel(r'$m.s^{-2}$')
            # axe.set_ylim([-2e-5,2e-5])
            axe.set_ylim([-5e-5,5e-5])
        fig.suptitle('U budget test set')
        fig.savefig(path_save_png+f'test_set_U_budget_slab_{NN_MODEL_NAME}.png')
    
    
    #################
    # Cloud of point
    #################
    # show U estimated vs true U
    dirr = 0
    
    # # 1 time step integration case
    # Ntraj = NN_train_data['forcing'].shape[0] -1 #[0,0:2,:,:]
    # _,_,ny,nx = NN_train_data['forcing'].shape
    # U_array = np.zeros((3, Ntraj, ny, nx))
    
    
    # for k in range(5):
    #     myforcing_for_this_traj = NN_train_data['forcing'][k:k+1]
        
    #     mytraj_NN = Integration_Euler(X0        = NN_train_data['forcing'][k,0:2,:,:], 
    #                                 forcing   = NN_train_data['forcing'][k,2:4,:,:][np.newaxis,:,:,:], 
    #                                 features  = NN_train_data['features'][k,:,:,:][np.newaxis,:,:,:], 
    #                                 RHS       = best_RHS_NN, 
    #                                 dt        = dt_Euler, 
    #                                 dt_forcing= dt_forcing,
    #                                 Nsteps    = 1,
    #                                 norms     = NN_norms_tr)[-1][dirr]
        
    #     mytraj_slab = Integration_Euler(X0        = slab_train_data['forcing'][k,0:2,:,:], 
    #                                 forcing   = slab_train_data['forcing'][k,2:4,:,:][np.newaxis,:,:,:], 
    #                                 features  = slab_train_data['features'][k,:,:,:][np.newaxis,:,:,:], 
    #                                 RHS       = best_RHS_slab, 
    #                                 dt        = dt_Euler, 
    #                                 dt_forcing= dt_forcing,
    #                                 Nsteps    = 1,
    #                                 norms     = slab_norms_tr)[-1][dirr]
        
    #     truth = NN_train_data['forcing'][k+1,0:2,:,:][dirr]
    #     print(mytraj_NN.shape)
    #     U_array[0,k] = mytraj_NN
    #     U_array[1,k] = mytraj_slab
    #     U_array[2,k] = truth
        
    # raise Exception    
    # xy_NN = np.vstack([U_array[-1].flatten(), U_array[0].flatten()])
    
    
    
    # kde_NN = gaussian_kde(U_array[0].flatten())
    # kde_truth = gaussian_kde(U_array[-1].flatten())
    # kde_slab = gaussian_kde(U_array[1].flatten())
    # range_u = np.arange(-0.1,0.1,0.001)
    # density_NN = kde_NN.pdf(range_u)
    # density_truth = kde_truth.pdf(range_u)
    # density_slab = kde_slab.pdf(range_u)
        
    # fig, ax = plt.subplots(1,1,figsize = (10,10), constrained_layout=True,dpi=100)
    # # ax.plot(range_u, density_NN, c='b', label='NN density')
    # # ax.plot(range_u, density_truth, c='k', label='truth density')
    # # ax.plot(range_u, density_slab, c='g', label='slab density')
    # ax.hist((U_array[1]-U_array[-1]).flatten(), bins=40, range=[-0.01,0.01])
    # ax.legend()
    # ax.set_xlabel('U')
    # ax.set_ylabel('hist of U estimate')
    
    
    # fig, ax = plt.subplots(1,1,figsize = (10,10), constrained_layout=True,dpi=100)
    # ax.scatter(U_array[-1], U_array[0], c='b', label='hybrid')
    # ax.scatter(U_array[-1], U_array[1], c='g', label='slab')
    # ax.plot(U_array[-1].flatten(), U_array[-1].flatten(), c='k')
    # ax.set_xlabel('U truth')
    # ax.set_ylabel('U estimated')
    # ax.legend()
    # ax.set_title('1 integration, train dataset')
    
    nt,_,ny,nx = NN_train_data['forcing'].shape
    U_array = jnp.zeros((3, nt-1, ny, nx))
    
    
    def fn_for_scan(carry, k):
        U_array = carry
        start = k*1
        
        forcing_for_this_traj = lax.dynamic_slice(NN_train_data['forcing'],    (start, 0, 0, 0),   (2, NN_train_data['forcing'].shape[1], ny, nx))
        features_for_this_traj = lax.dynamic_slice(NN_train_data['features'],  (start, 0, 0, 0),   (2, NN_train_data['features'].shape[1], ny, nx))
        mytraj_NN = Integration_Euler(X0      = forcing_for_this_traj[0,0:2,:,:], 
                                    forcing   = forcing_for_this_traj[:,2:4,:,:], 
                                    features  = features_for_this_traj, 
                                    RHS       = best_RHS_NN, 
                                    dt        = dt_Euler, 
                                    dt_forcing= dt_forcing,
                                    Nsteps    = 1,
                                    norms     = NN_norms_tr)[-1][dirr]
        
        forcing_for_this_traj = lax.dynamic_slice(slab_train_data['forcing'],    (start, 0, 0, 0),   (2, NN_train_data['forcing'].shape[1], ny, nx))
        features_for_this_traj = lax.dynamic_slice(slab_train_data['features'],  (start, 0, 0, 0),   (2, NN_train_data['features'].shape[1], ny, nx))
        mytraj_slab = Integration_Euler(X0    = forcing_for_this_traj[0,0:2,:,:], 
                                    forcing   = forcing_for_this_traj[:,2:4,:,:],  
                                    features  = features_for_this_traj, 
                                    RHS       = best_RHS_slab, 
                                    dt        = dt_Euler, 
                                    dt_forcing= dt_forcing,
                                    Nsteps    = 1,
                                    norms     = slab_norms_tr)[-1][dirr]
         
        U_array = U_array.at[0,k].set(mytraj_NN)
        U_array = U_array.at[1,k].set(mytraj_slab)
        U_array = U_array.at[2,k].set(forcing_for_this_traj[1,0:2,:,:][dirr])

        return U_array, None    
    final, _ = lax.scan(fn_for_scan, U_array, xs=np.arange(1,nt))
    U_array = final
    
    

    
    print(U_array.shape)
    
    #truth = NN_train_data['forcing'][1:,0:2,:,:][:,dirr,:,:]
    # U_array = np.zeros((3, nt-1, ny, nx))
    # for k in range(1,nt):
    #     print(k)
    #     mytraj_NN = Integration_Euler(X0        = NN_train_data['forcing'][k-1,0:2,:,:], 
    #                                 forcing   = NN_train_data['forcing'][k-1,2:4,:,:][np.newaxis,:,:,:], 
    #                                 features  = NN_train_data['features'][k-1,:,:,:][np.newaxis,:,:,:], 
    #                                 RHS       = best_RHS_NN, 
    #                                 dt        = dt_Euler, 
    #                                 dt_forcing= dt_forcing,
    #                                 Nsteps    = 1,
    #                                 norms     = NN_norms_tr)[-1][dirr]
    
    #     mytraj_slab = Integration_Euler(X0        = slab_train_data['forcing'][k-1,0:2,:,:], 
    #                                 forcing   = slab_train_data['forcing'][k-1,2:4,:,:][np.newaxis,:,:,:], 
    #                                 features  = slab_train_data['features'][k-1,:,:,:][np.newaxis,:,:,:], 
    #                                 RHS       = best_RHS_slab, 
    #                                 dt        = dt_Euler, 
    #                                 dt_forcing= dt_forcing,
    #                                 Nsteps    = 1,
    #                                 norms     = slab_norms_tr)[-1][dirr]
    #     truth = NN_train_data['forcing'][k,0:2,:,:][dirr]
    #     U_array[0,k] = mytraj_NN
    #     U_array[1,k] = mytraj_slab
    #     U_array[2,k] = truth
    
    
    xtime = np.arange(1,nt)*onehour/oneday
    fig, ax = plt.subplots(1,1,figsize = (7,4), constrained_layout=True,dpi=100)
    ax.plot(xtime, U_array[0].mean(axis=(1,2)), c='b', label='hybrid')
    ax.plot(xtime, U_array[1].mean(axis=(1,2)), c='r', label='slab', ls='--')
    ax.plot(xtime, U_array[2].mean(axis=(1,2)), c='k', label='truth')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('U ageo (m/s)')
    ax.legend()
    
    
    """Plot: trajectory reconstructed on train dataset (and next on test data set)
    
        -> solution K0 and r
        -> solution K0 and MLP_linear(U,V)
        -> solution K0 and non linear NN (CNN ?), using only  U and V as features
    """
    
    """Plot: comparison of the dissipation term with different parametrizations, same models as above
    """
    
    """Plot: accuracy of the reconstruction with respect to the length of integration time, from offline to full online
    """
    
    
plt.show()

