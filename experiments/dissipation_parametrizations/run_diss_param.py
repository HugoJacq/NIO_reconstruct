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

# jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax

#====================================================================================================
# PARAMETERS
#====================================================================================================

TRAIN           = False      # run the training and save best model
SAVE            = True      # save model and figures
PLOTTING        = True      # plot figures

MAX_STEP        = 100           # number of epochs
PRINT_EVERY     = 10            # print infos every 'PRINT_EVERY' epochs
BATCH_SIZE      = 1000          # size of a batch (time), set to -1 for no batching
SEED            = 5678          # for reproductibility
features_names  = ['U','V']     # what features to use in the NN


# Defining data
test_ratio  = 20            # % of hours for test (over the data)
Nsize       = 128           # size of the domain, in nx ny
dt_forcing  = 3600          # time step of forcing
dx          = 0.1           # X data resolution in ° 
dy          = 0.1           # Y data resolution in ° 
K0          = 1e-6           # initial K0










#====================================================================================================

#====================================================================================================

# create folder for each set of features
name_base_folder = 'features'+''.join('_'+variable for variable in features_names)+'/'
os.system(f'mkdir -p {name_base_folder}')

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
Ntests = test_ratio*len(ds.time)//100 # how many instants used for test

# NN initialization
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
""" here add model initialization"""


# make test and train datasets
"""add: data_maker: target, forcing, features""" 
"""add: data iterator"""

# add path to save figures according to model name
path_save = name_base_folder+'namemodel'+'/'
os.system(f'mkdir -p {path_save}')

if TRAIN:
    
    """optimizer initialization"""
    """optionial: learning rate scheduler initialization"""
    
    # train loop
    """code: training loop"""
    
    if SAVE:
        best_model='toto'
        eqx.tree_serialise_leaves(path_save+'best_model.pt',best_model)
        
    """Plot: Loss(epochs) for train and test datasets"""
    print('done!')
    
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