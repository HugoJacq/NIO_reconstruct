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

# my imports
from train_preprocessing import data_maker, batch_loader
from train_functions import train
from constants import oneday, distance_1deg_equator
from models_definition import RHS, Dissipation_Rayleigh, Stress_term, Coriolis_term
#====================================================================================================
# PARAMETERS
#====================================================================================================

TRAIN           = True      # run the training and save best model
SAVE            = True      # save model and figures
PLOTTING        = True      # plot figures

MAX_STEP        = 200           # number of epochs
PRINT_EVERY     = 10            # print infos every 'PRINT_EVERY' epochs
BATCH_SIZE      = -1          # size of a batch (time), set to -1 for no batching
SEED            = 5678          # for reproductibility
features_names  = ['U','V']     # what features to use in the NN
forcing_names   = []            # U,V,TAx,TAy already in by default
N_integration_steps = 1         # 1 for offline, more for online

# Defining data
test_ratio  = 20            # % of hours for test (over the data)
Nsize       = 128           # size of the domain, in nx ny
dt_forcing  = 3600          # time step of forcing
dx          = 0.1           # X data resolution in ° 
dy          = 0.1           # Y data resolution in ° 
K0          = 1e-6          # initial K0
dt_Euler    = 60.           # secondes









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


# warnings
if N_integration_steps > BATCH_SIZE and BATCH_SIZE>0:
    print(f'You have chosen to do online training but the number of integration step ({N_integration_steps}) is greater than the batch_size ({BATCH_SIZE})')
    print(f'N_integration_steps has been reduced to the batch size value ({BATCH_SIZE})')
    N_integration_steps = BATCH_SIZE
if N_integration_steps<0:
    print(f'N_integration_steps < 0, N_integration_steps = BATCH_SIZE ({BATCH_SIZE})')
    N_integration_steps = BATCH_SIZE

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
myDissipation = Dissipation_Rayleigh(R = 0.01, to_train=True)
myRHS = RHS(myCoriolis, myStress, myDissipation)

# make test and train datasets
train_data, test_data = data_maker(ds=ds,
                                    test_ratio=test_ratio, 
                                    features_names=['U','V'],
                                    forcing_names=[],
                                    dx=dx*distance_1deg_equator,
                                    dy=dy*distance_1deg_equator)

train_iterator = batch_loader(data_set=train_data,
                            batch_size=BATCH_SIZE)


# add path to save figures according to model name
path_save = name_base_folder+model_name+'/'
os.system(f'mkdir -p {path_save}')

if TRAIN:
    
    optimizer = optax.adam(1e-2)
    """optional: learning rate scheduler initialization"""

    # train loop
    lastmodel, bestmodel, Train_loss, Test_loss, opt_state_save = train(
                                                                the_model   = myRHS,
                                                                optim       = optimizer,
                                                                iter_train_data = train_iterator,
                                                                test_data       = test_data,
                                                                maxstep    = MAX_STEP,
                                                                tol         = 1e-5)
    if SAVE:
        eqx.tree_serialise_leaves(path_save+f'best_{model_name}.pt', bestmodel)
        
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

