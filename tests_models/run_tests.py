"""
Here a script that tests the models from the folder "models"
"""


import numpy as np
import time as clock
import matplotlib.pyplot as plt
import sys
import os
import xarray as xr
sys.path.insert(0, '../src')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
import jax
import jax.numpy as jnp

from models.classic_slab import jslab, jslab_Ue_Unio, jslab_kt, jslab_kt_2D, jslab_rxry, kt_ini, kt_1D_to_2D, pkt2Kt_matrix
import forcing
import inv
import observations
from tests_functions import run_forward_cost_grad, plot_traj_1D, plot_traj_2D
import tools
from constants import *

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
t0                  = 240*oneday
t1                  = 300*oneday
dt                  = 60.        # timestep of the model (s) 

# What to test
FORWARD_PASS        = False      # tests forward, cost, gradcost
MINIMIZE            = True      # switch to do the minimisation process
maxiter             = 50         # max number of iteration
PLOT_TRAJ           = True

# Switches
TEST_SLAB                   = False
TEST_SLAB_KT                = False
TEST_SLAB_KT_FILTERED_FC    = False
TEST_SLAB_KT_2D             = False
TEST_SLAB_RXRY              = False # WIP
TEST_SLAB_Ue_Unio           = True

# PLOT
dpi=200
path_save_png = './png_tests_models/'

# =================================
# Forcing, OSSE and observations
# =================================
# 1D
point_loc = [-50.,35.]
#point_loc = [-50.,46.] # should have more NIOs ?
point_loc = [-70., 35.]
# 2D
R = 10.0 # 20°x20° -> ~6.5Go of VRAM for grad
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
# name_regrid = ['croco_1h_inst_surf_2005-01-01-2005-01-31_0.1deg_conservative.nc',
#               'croco_1h_inst_surf_2005-02-01-2005-02-28_0.1deg_conservative.nc',
#               'croco_1h_inst_surf_2005-03-01-2005-03-31_0.1deg_conservative.nc',
#               'croco_1h_inst_surf_2005-04-01-2005-04-30_0.1deg_conservative.nc',
#               'croco_1h_inst_surf_2005-05-01-2005-05-31_0.1deg_conservative.nc',
#               'croco_1h_inst_surf_2005-06-01-2005-06-30_0.1deg_conservative.nc',]
#name_regrid = ['croco_1h_inst_surf_2005-01-01-2005-01-31_0.1deg_conservative.nc']

# Observations
period_obs          = oneday #86400      # s, how many second between observations  

# ============================================================
# END PARAMETERS
# ============================================================

namesave_loc = str(point_loc[0])+'E_'+str(point_loc[1])+'N'
namesave_loc_area = str(point_loc[0])+'E_'+str(point_loc[1])+'N_R'+str(R)
os.system('mkdir -p '+path_save_png)

if __name__ == "__main__": 
    
    file = []
    for ifile in range(len(name_regrid)):
        file.append(path_regrid+name_regrid[ifile])
    #file = path_regrid+name_regrid 
    if TEST_SLAB or TEST_SLAB_KT or TEST_SLAB_KT_FILTERED_FC or TEST_SLAB_RXRY or TEST_SLAB_Ue_Unio:
        forcing1D = forcing.Forcing1D(point_loc, t0, t1, dt_forcing, file)
        observations1D = observations.Observation1D(point_loc, period_obs, t0, t1, dt_OSSE, file)
    if TEST_SLAB_KT_2D:
        forcing2D = forcing.Forcing2D(dt_forcing, t0, t1, file, LON_bounds, LAT_bounds)
        observations2D = observations.Observation2D(period_obs, t0, t1, dt_OSSE, file, LON_bounds, LAT_bounds)
    
    ### WARNINGS
    dsfull = xr.open_mfdataset(file)
    # warning about t1>length of forcing
    if t1 > len(dsfull.time)*dt_forcing:
        print(f'You chose to run the model for t1={t1//oneday} days but the forcing is available up to t={len(dsfull.time)*dt_forcing//oneday} days\n'
                        +f'I will use t1={len(dsfull.time)*dt_forcing//oneday} days')
        t1 = len(dsfull.time)*dt_forcing
    # warning about 2D selection
    minlon, maxlon = np.amin(dsfull.lon), np.amax(dsfull.lon)
    minlat, maxlat = np.amin(dsfull.lat), np.amax(dsfull.lat)
    #print(minlon.values,maxlon.values,minlat.values,maxlat.values) # = -81.95 -36.05 22.55 48.75
    if (minlon + R > point_loc[0]) or (point_loc[0] > maxlon - R):
        raise Exception(f"your choice of LON in 'point_loc'({point_loc}) and 'R'({R}) is outside of the domain, please retry")
    if (minlat + R > point_loc[1]) or (point_loc[1] > maxlat - R):
        raise Exception(f"your choice of LAT in 'point_loc'({point_loc}) and 'R'({R}) is outside of the domain, please retry")
    ### END WARNINGS
    
    
    
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
        var_dfx = inv.Variational(mymodel,observations1D)
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'jslab_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)
        
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
                           
        name_save = 'jslab_kt_'+namesave_loc
        if PLOT_TRAJ: 
            plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)
    
    if TEST_SLAB_KT_FILTERED_FC:
        print('* test jslab_kt with filter at fc before computing cost')
        # control vector
        pk = jnp.asarray([-11.31980127, -10.28525189])   
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        
        # pk = jnp.asarray([-11.31980127, -11.31980127, -11.31980127, -11.31980127, -11.31980127, # dTK = 3*86400
        #                 -11.31980127, -11.31980127, -11.31980127, -11.31980127, -10.28525189,
        #                 -10.28525189, -10.28525189, -10.28525189, -10.28525189, -10.28525189,
        #                 -10.28525189, -10.28525189, -10.28525189,])
        
        # parameters
        TAx = jnp.asarray(forcing1D.TAx)
        TAy = jnp.asarray(forcing1D.TAy)
        fc = jnp.asarray(forcing1D.fc)
        
        call_args = t0, t1, dt
        mymodel = jslab_kt(pk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args, k_base=k_base)
        var_dfx = inv.Variational(mymodel,observations1D, filter_at_fc=True)
        
        # dynamic_model, static_model = var_dfx.my_partition(mymodel)
        # for _ in range(10):
        #     time1 = clock.time()
        #     _ = mymodel() # call_args
        #     print(' time, forward model (with compile)',clock.time()-time1)
            
            # time6 = clock.time()
            # _, _ = var_dfx.grad_cost(dynamic_model, static_model)
            # print(' time, gradcost',clock.time()-time6)
    
    
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, gtol=1e-5, verbose=True)   
            print(' time, minimize',clock.time()-t7)
          
        
        name_save = 'jslab_kt_'+namesave_loc
        if PLOT_TRAJ:           
            plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)
        
    if TEST_SLAB_KT_2D:
        #with jax.profiler.trace("/tmp/tensorboard"):
        #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        print('* test jslab_kt_2D')
        # control vector
        pk = jnp.asarray([-11.31980127, -10.28525189])   
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        
        # parameters
        TAx = jnp.asarray(forcing2D.TAx)
        TAy = jnp.asarray(forcing2D.TAy)
        fc = jnp.asarray(forcing2D.fc)
        
        call_args = t0, t1, dt
        mymodel = jslab_kt_2D(pk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args,use_difx=False, k_base=k_base)
        var_dfx = inv.Variational(mymodel,observations2D)
        
    
        
        # Run the operations to be profiled
        # U,V = mymodel()
        # U.block_until_ready()
        # V.block_until_ready()
    
        #jax.profiler.save_device_memory_profile("memory_R"+str(R)+".prof")
        # pprof --svg memory.prof

        
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'jslab_kt_2D_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_2D(mymodel, var_dfx, forcing2D, observations2D, name_save, point_loc, LON_bounds, LAT_bounds, path_save_png, dpi) 
    
    if TEST_SLAB_RXRY:
        print('* test jslab_rxry')
        # control vector
        pk = jnp.asarray([-11.,-10.,-10.])    
        
        # parameters
        TAx = jnp.asarray(forcing1D.TAx)
        TAy = jnp.asarray(forcing1D.TAy)
        fc = jnp.asarray(forcing1D.fc)
        
        call_args = t0, t1, dt
        
        #mymodel = jslab(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode)
        mymodel = jslab_rxry(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args)
        var_dfx = inv.Variational(mymodel,observations1D)
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'jslab_rxry_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)
    
    if TEST_SLAB_Ue_Unio:
        print('* test jslab_Ue_Unio')
        # control vector
        pk = jnp.asarray([-11.,-10.,-10.,-9])    
        
        # parameters
        TA = forcing1D.TAx + 1j*forcing1D.TAy
        TAx_f,TAy_f = tools.my_fc_filter( dt_forcing, TA, forcing1D.fc)
        TAx = jnp.asarray(TAx_f)
        TAy = jnp.asarray(TAy_f)
        fc = jnp.asarray(forcing1D.fc)
        
        call_args = t0, t1, dt
        
        #mymodel = jslab(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode)
        mymodel = jslab_Ue_Unio(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args)
        var_dfx = inv.Variational(mymodel,observations1D)
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'jslab_rxry_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)
    
    end = clock.time()
    print('Total execution time = '+str(jnp.round(end-start,2))+' s')
    plt.show()