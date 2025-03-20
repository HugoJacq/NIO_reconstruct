"""
Here a script that tests the models from the folder "models"
"""

import jax.numpy as jnp
import numpy as np
import time as clock
import matplotlib.pyplot as plt
import sys
import os
#sys.path.append('..')
sys.path.insert(0, '../src')

from models.jslab import jslab,jslab_kt
import forcing
import inv
import observations
from tests_functions import run_forward_cost_grad, plot_traj
import tools

start = clock.time()

# ============================================================
# PARAMETERS
# ============================================================
ON_HPC      = False      # on HPC

# model parameters
Nl                  = 1         # number of layers for multilayer models
dTK                 = 3*86400   # how much vectork K changes with time, basis change to exp
AD_mode             = 'F'       # forward mode for AD 

# run parameters
t0 = 0.
t1 = 28*86400. #
dt                  = 60.        # timestep of the model (s) 

# Minimizer
MINIMIZE            = True      # switch to do the minimisation process
maxiter             = 100       # max number of iteration

# Switches
TEST_SLAB           = True
TEST_SLAB_KT        = False
TEST_SLAB_KT_FILTERED_FC = True

# PLOT
dpi=200
path_save_png = './png_tests_models/'

# =================================
# Forcing, OSSE and observations
# =================================
# 1D
point_loc = [-50.,35.]
# 2D
R = 0.2 # °
LON_bounds = [point_loc[0]-R,point_loc[0]+R]
LAT_bounds = [point_loc[1]-R,point_loc[1]+R]

# Forcing
dt_forcing          = 3600      # forcing timestep

# OSSE from Croco
dt_OSSE             = 3600      # timestep of the OSSE (s)
path_regrid = '../data_regrid/'
name_regrid = 'croco_1h_inst_surf_2006-02-01-2006-02-28_0.1deg_conservative.nc'

# Observations
period_obs          = 86400      # s, how many second between observations  

# ============================================================
# END PARAMETERS
# ============================================================

namesave_loc = str(point_loc[0])+'E_'+str(point_loc[1])+'N'
namesave_loc_area = str(point_loc[0])+'E_'+str(point_loc[1])+'N_R'+str(R)
os.system('mkdir -p '+path_save_png)

if __name__ == "__main__": 
    
    
    file = path_regrid+name_regrid 
    forcing1D = forcing.Forcing1D(point_loc, dt_OSSE, file)
    observations1D = observations.Observation1D(point_loc, period_obs, dt_OSSE, file)
    # forcing2D = forcing.Forcing2D(dt_forcing, file, LON_bounds, LAT_bounds)
    # observations2D = observations.Observation2D(period_obs, dt_OSSE, file, LON_bounds, LAT_bounds)
    
    
    
    if TEST_SLAB:
        print('* test jslab')
        # control vector
        pk = jnp.asarray([-11.31980127, -10.28525189])    
        
        # parameters
        TAx = jnp.asarray(forcing1D.TAx)
        TAy = jnp.asarray(forcing1D.TAy)
        fc = jnp.asarray(forcing1D.fc)
        
        call_args = t0, t1, dt
        
        #mymodel = jslab(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode)
        mymodel = jslab(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args)
        var_dfx = inv.Variational_diffrax(mymodel,observations1D)
        
        run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'jslab_'+namesave_loc
        plot_traj(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)
        
    if TEST_SLAB_KT:
        print('* test jslab_kt')
        # control vector
        pk = jnp.asarray([-11.31980127, -10.28525189])    
        
        # parameters
        TAx = jnp.asarray(forcing1D.TAx)
        TAy = jnp.asarray(forcing1D.TAy)
        fc = jnp.asarray(forcing1D.fc)
        
        call_args = t0, t1, dt
        #NdT = int((t1-t0)//dTK) # jnp.array(int((t1-t0)//dTK))
        mymodel = jslab_kt(pk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args)
        var_dfx = inv.Variational_diffrax(mymodel,observations1D)
        
        run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'jslab_kt_'+namesave_loc
        plot_traj(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)
    
    if TEST_SLAB_KT_FILTERED_FC:
        print('* test jslab_kt with filter at fc before computing cost')
        # control vector
        pk = jnp.asarray([-11.31980127, -10.28525189])    
        
        # parameters
        TAx = jnp.asarray(forcing1D.TAx)
        TAy = jnp.asarray(forcing1D.TAy)
        fc = jnp.asarray(forcing1D.fc)
        
        call_args = t0, t1, dt
        mymodel = jslab_kt(pk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args)
        var_dfx = inv.Variational_diffrax(mymodel,observations1D, filter_at_fc=True)
        
        run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, gtol=1e-6, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'jslab_kt_'+namesave_loc
        plot_traj(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)
        
    end = clock.time()
    print('Total execution time = '+str(jnp.round(end-start,2))+' s')
    plt.show()