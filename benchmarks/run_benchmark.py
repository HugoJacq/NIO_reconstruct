"""
Here we want to find the optimal dTK for the vector pk.
"""


import numpy as np
import time as clock
import matplotlib.pyplot as plt
import sys
import os
import xarray as xr
sys.path.insert(0, '../src')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
import jax.numpy as jnp

from models.classic_slab import jslab, jslab_kt, jslab_kt_2D, kt_ini, kt_1D_to_2D, pkt2Kt_matrix
import forcing
import inv
import observations
#from tests_functions import run_forward_cost_grad, plot_traj_1D, plot_traj_2D
from constants import *

from functions_benchmark import benchmark_all


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
t0                  = 200*oneday
t1                  = 300*oneday
dt                  = 60.        # timestep of the model (s) 

# What to test
FORWARD_PASS        = False      # tests forward, cost, gradcost
MINIMIZE            = True      # switch to do the minimisation process
maxiter             = 2         # max number of iteration
PLOT_TRAJ           = True

L_MODELS_TO_BENCHMARK = ['']

# PLOT
dpi=200
path_save_png = './png_benchmark/'

# =================================
# Forcing, OSSE and observations
# =================================
# 1D
point_loc = [-50.,35.]
#point_loc = [-50.,46.] # should have more NIOs ?
point_loc = [-70., 35.]
# 2D
R = 5.0 # 20°x20° -> ~6.5Go of VRAM for grad
LON_bounds = [point_loc[0]-R,point_loc[0]+R]
LAT_bounds = [point_loc[1]-R,point_loc[1]+R]
# Forcing
dt_forcing          = onehour      # forcing timestep

# OSSE from Croco
dt_OSSE             = onehour      # timestep of the OSSE (s)
path_regrid = '../data_regrid/'
name_regrid = ['croco_1h_inst_surf_2005-01-01-2005-01-31_0.1deg_conservative.nc',
              'croco_1h_inst_surf_2005-02-01-2005-02-28_0.1deg_conservative.nc',
              'croco_1h_inst_surf_2005-03-01-2005-03-31_0.1deg_conservative.nc',
              'croco_1h_inst_surf_2005-04-01-2005-04-30_0.1deg_conservative.nc',
              'croco_1h_inst_surf_2005-05-01-2005-05-31_0.1deg_conservative.nc',
              'croco_1h_inst_surf_2005-06-01-2005-06-30_0.1deg_conservative.nc',
              'croco_1h_inst_surf_2005-07-01-2005-07-31_0.1deg_conservative.nc',
              'croco_1h_inst_surf_2005-08-01-2005-08-31_0.1deg_conservative.nc',
              'croco_1h_inst_surf_2005-09-01-2005-09-30_0.1deg_conservative.nc',
              'croco_1h_inst_surf_2005-10-01-2005-10-31_0.1deg_conservative.nc',
              'croco_1h_inst_surf_2005-11-01-2005-11-30_0.1deg_conservative.nc',
              'croco_1h_inst_surf_2005-12-01-2005-12-31_0.1deg_conservative.nc']

# Observations
period_obs          = oneday #86400      # s, how many second between observations  

# ============================================================
# END PARAMETERS
# ============================================================

os.system('mkdir -p '+path_save_png)

if __name__ == "__main__": 
    
    file = []
    for ifile in range(len(name_regrid)):
        file.append(path_regrid+name_regrid[ifile])
        
    ### WARNINGS
    dsfull = xr.open_mfdataset(file)
    # warning about t1>length of forcing
    if t1 > len(dsfull.time)*dt_forcing:
        print(f'You chose to run the model for t1={t1//oneday} days but the forcing is available up to t={len(dsfull.time)*dt_forcing//oneday} days\n'
                        +f'I will use t1={len(dsfull.time)*dt_forcing//oneday} days')
        t1 = len(dsfull.time)*dt_forcing
    ### END WARNINGS
    
    
    forcing1D = forcing.Forcing_from_PAPA(dt_forcing, t0, t1, file)
    observations1D = observations.Observation_from_PAPA(period_obs, t0, t1, dt_forcing, file)
    forcing2D = forcing.Forcing2D(dt_forcing, t0, t1, file, LON_bounds, LAT_bounds)
    observations2D = observations.Observation2D(period_obs, t0, t1, dt_OSSE, file, LON_bounds, LAT_bounds)
        
    print('My slab at station PAPA')
    # control vector
    pk = jnp.asarray([-11.31980127, -10.28525189])   
        
    # parameters
    TAx = jnp.asarray(forcing1D.TAx)
    TAy = jnp.asarray(forcing1D.TAy)
    fc = jnp.asarray(forcing1D.fc)
    
    call_args = t0, t1, dt
    
    
    L_models = []
    L_obs = []
    for nmodel in L_MODELS_TO_BENCHMARK:

        if nmodel in ['']:
            "ini kt"
            NdT = len(np.arange(t0, t1,dTK))
            pk = kt_ini(pk, NdT)
            
        if nmodel in ['']:
            "2D"
            frc, obs = forcing2D, observations2D
        else:
            "1D"
            frc, obs = forcing1D, observations1D
            
        if nmodel=='jslab_kt':
            'model selector, 1 for each model'
            mymodel = jslab_kt(pk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args, k_base=k_base)
        elif nmodel=='jslab':
            mymodel= jslab(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args)
                    
        L_models.append(mymodel)
        L_obs.append(obs)
    
    benchmark_all(L_models, L_obs, Nexec=10)