"""
Here we want to find the optimal dTK for the vector pk.
"""


import numpy as np
import time as clock
import matplotlib.pyplot as plt
import sys
import os
import xarray as xr
sys.path.insert(0, '../../src')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
import jax
import jax.numpy as jnp
import cdflib  # to read cdf files

from models.classic_slab import jslab, jslab_kt, jslab_kt_2D, kt_ini, kt_1D_to_2D, pkt2Kt_matrix
import forcing
import inv
import observations
#from tests_functions import run_forward_cost_grad, plot_traj_1D, plot_traj_2D
import tools
from constants import *

sys.path.insert(0, '../../tests_models')
from tests_functions import *

start = clock.time()



# ============================================================
# PARAMETERS
# ============================================================
#ON_HPC      = False      # on HPC

# model parameters
Nl                  = 1         # number of layers for multilayer models
dTK                 = 10*oneday   # how much vectork K changes with time, basis change to exp
k_base              = 'gauss'   # base of K transform. 'gauss' or 'id'
AD_mode             = 'F'       # forward mode for AD 

# run parameters
t0                  = 0*oneday
t1                  = 365*oneday
dt                  = 60.        # timestep of the model (s) 

# What to test
FORWARD_PASS        = False      # tests forward, cost, gradcost
MINIMIZE            = True      # switch to do the minimisation process
maxiter             = 100         # max number of iteration
PLOT_TRAJ           = True



TEST_SLAB_KT                = False

# PLOT
dpi=200
path_save_png = './png_models_at_PAPA/'

# =================================
# Forcing, OSSE and observations
# =================================

# PAPA station is located at 50.1°N, 144.9°W
point_loc = [-144.9, 50.1]

# Forcing : PAPA data is hourly 
dt_forcing          = onehour      # forcing timestep

path_data = '../../data_PAPA_2018/'
name_data = ['cur50n145w_hr.nc',
              'd50n145w_hr.nc',
              'lw50n145w_hr.nc',
              'rad50n145w_hr.nc',
              's50n145w_hr.nc',
              'sss50n145w_hr.nc',
              'sst50n145w_hr.nc',
              't50n145w_hr.nc',
              'w50n145w_hr.nc']

# Observations
period_obs          = oneday # 86400      # s, how many second between observations  

# ============================================================
# END PARAMETERS
# ============================================================

os.system('mkdir -p '+path_save_png)

if __name__ == "__main__": 
    
    file = []
    for ifile in range(len(name_data)):
        file.append(path_data+name_data[ifile])
        
    ### WARNINGS
    dsfull = xr.open_mfdataset(file)
    # warning about t1>length of forcing
    if t1 > len(dsfull.time)*dt_forcing:
        print(f'You chose to run the model for t1={t1//oneday} days but the forcing is available up to t={len(dsfull.time)*dt_forcing//oneday} days\n'
                        +f'I will use t1={len(dsfull.time)*dt_forcing//oneday} days')
        t1 = len(dsfull.time)*dt_forcing
    ### END WARNINGS
    
    
    forcing1D = forcing.Forcing_from_PAPA(t0, t1, dt_forcing, file)
    observations1D = observations.Observation_from_PAPA(period_obs, t0, t1, dt_forcing, file)
    
    raise Exception
    
    if TEST_SLAB_KT:
        print('* test jslab_kt')
        # control vector
        pk = jnp.asarray([-11.31980127, -10.28525189])   
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        
        # pk = jnp.asarray([-11.31980127, -11.31980127, -11.31980127, -11.31980127, -11.31980127,
        #                 -11.31980127, -11.31980127, -11.31980127, -11.31980127, -10.28525189,
        #                 -10.28525189, -10.28525189, -10.28525189, -10.28525189, -10.28525189,
        #                 -10.28525189, -10.28525189, -10.28525189,])  
        
        # parameters
        TAx = jnp.asarray(forcing1D.TAx)
        TAy = jnp.asarray(forcing1D.TAy)
        fc = jnp.asarray(forcing1D.fc)
        
        call_args = t0, t1, dt
        #NdT = int((t1-t0)//dTK) # jnp.array(int((t1-t0)//dTK))
        mymodel = jslab_kt(pk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args, k_base=k_base)
        var_dfx = inv.Variational(mymodel,observations1D)
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'jslab_kt_'
        if PLOT_TRAJ: 
            plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)