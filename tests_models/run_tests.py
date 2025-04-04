"""
Here a script that tests the models from the folder "models"
"""


import numpy as np
import time as clock
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qtagg')
import sys
import os
import xarray as xr
sys.path.insert(0, '../src')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
import jax.numpy as jnp
import equinox as eqx
#import jax
#jax.config.update('jax_platform_name', 'cpu')

# my inmports
from models.classic_slab import jslab, jslab_Ue_Unio, jslab_kt, jslab_kt_2D, jslab_kt_2D_adv_Ut, jslab_rxry, jslab_kt_Ue_Unio, jslab_kt_2D_adv
from models.unsteak import junsteak, junsteak_kt, junsteak_kt_2D
from basis import kt_ini

import forcing
import inv
import observations
from tests_functions import run_forward_cost_grad, plot_traj_1D, plot_traj_2D, idealized_run
import tools
from constants import *

start = clock.time()

# ============================================================
# PARAMETERS
# ============================================================
#ON_HPC      = False      # on HPC

# model parameters
Nl                  = 2             # number of layers for multilayer models
dTK                 = 40*oneday     # how much vectork K changes with time, basis change to 'k_base'
k_base              = 'gauss'       # base of K transform. 'gauss' or 'id'
AD_mode             = 'F'           # forward mode for AD (for diffrax' diffeqsolve)

# run parameters
t0                  = 60*oneday    # start day 
t1                  = 100*oneday    # end day
dt                  = 60.           # timestep of the model (s) 

# What to test
PLOT_TRAJ           = True      # Show a trajectory
FORWARD_PASS        = True      # How fast the model is running ?
MINIMIZE            = True      # Does the model converges to a solution ?
maxiter             = 50        # if MINIMIZE: max number of iteration


ON_PAPA             = False      # use PAPA station data, only for 1D models
FILTER_AT_FC        = False      # minimize filtered ageo current with obs if model has this option
IDEALIZED_RUN       = False       # try the model on a step wind stress

# Switches
# slab based models
TEST_SLAB                   = False
TEST_SLAB_KT                = False
TEST_SLAB_KT_FILTERED_FC    = False
TEST_SLAB_KT_2D             = False
TEST_SLAB_RXRY              = False # WIP
TEST_SLAB_Ue_Unio           = False
TEST_SLAB_KT_Ue_Unio        = False # WIP
TEST_SLAB_KT_2D_ADV         = False # WIP, crash
TEST_SLAB_KT_2D_ADV_UT      = False
# fourier solving
TEST_SLAB_FFT               = False
# unsteak based
TEST_UNSTEAK                = False
TEST_UNSTEAK_KT             = False
TEST_UNSTEAK_KT_2D          = True

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
point_loc = [-50., 40.]
point_loc = [-46., 40.] # wind gust from early january 2018
point_loc = [-55., 37.5] # february 13th
point_loc = [-47.4,34.6] # march 8th 0300, t0=60 t1=100
# 2D
R = 1. # 5.0 
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
#name_regrid = ['croco_1h_inst_surf_2006-02-01-2006-02-28_0.1deg_conservative.nc']

path_papa = '../data_PAPA_2018/'
name_papa = ['cur50n145w_hr.nc',
              'd50n145w_hr.nc',
              'lw50n145w_hr.nc',
              'rad50n145w_hr.nc',
              's50n145w_hr.nc',
              'sss50n145w_hr.nc',
              'sst50n145w_hr.nc',
              't50n145w_hr.nc',
              'w50n145w_hr.nc']

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
        
    file_papa = []
    for ifile in range(len(name_papa)):
        file_papa.append(path_papa+name_papa[ifile])
    
    #file = path_regrid+name_regrid 
    if ON_PAPA:
        forcing1D = forcing.Forcing_from_PAPA(dt_forcing, t0, t1, file_papa)
        observations1D = observations.Observation_from_PAPA(period_obs, t0, t1, dt_forcing, file_papa)
    else:
        forcing1D = forcing.Forcing1D(point_loc, t0, t1, dt_forcing, file)
        observations1D = observations.Observation1D(point_loc, period_obs, t0, t1, dt_OSSE, file)
    if (TEST_SLAB_KT_2D or 
        TEST_SLAB_KT_2D_ADV or 
        TEST_SLAB_KT_2D_ADV_UT or
        TEST_UNSTEAK_KT_2D):
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
    ### END WARNINGSM
    
    
    if not ON_PAPA:
        indx = tools.nearest(dsfull.lon.values,point_loc[0])
        indy = tools.nearest(dsfull.lat.values,point_loc[1])
        txt_location = f'{dsfull.lon.values[indx]}°E, {dsfull.lat.values[indy]}°N'
    else:
        txt_location = 'PAPA station'
    print('Running: tests_models.py')   
    print('**************')
    print('Location is '+txt_location)
    print(f'2D slice is {LON_bounds}°E {LAT_bounds}°N')
    print(f'if multi layer, nl={Nl}')
    print('**************\n')
    
    if TEST_SLAB:
        print('* test jslab')
        # control vector
        pk = jnp.asarray([-11.31980127, -10.28525189])    
        
        # parameters
        TAx = jnp.asarray(forcing1D.TAx)
        TAy = jnp.asarray(forcing1D.TAy)
        fc = jnp.asarray(forcing1D.fc)
        
        call_args = t0, t1, dt
        
        mymodel = jslab(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args) # 
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
        pk = jnp.asarray([-11.,-10.])    
        
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
                           
        name_save = 'jslab_Ue_Unio_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)
    
    if TEST_SLAB_KT_Ue_Unio:
        print('* test jslab_kt_Ue_Unio')
        # control vector
        #pk = jnp.asarray([-11.,-10.,-10.,-9])   
        pk = jnp.asarray([-11.,-10.]) 
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        
        # pk = [-10.47863316,  -7.18847464, -13.05688585,  -9.79829942,  -7.78314267,
        #     -12.61145052, -15.2564004,  -18.29705017, -14.27925646,  -7.89936186,
        #     -16.13136193, -17.01115649,  -8.4708025,   -7.86208743, -14.81096965,
        #     -14.26312746,  -9.84993765, -14.21031189, -14.27335427, -14.00538017,
        #     -14.71485828, -14.51823757, -11.72167463, -15.92873515, -11.02885557,
        #     -12.13280427, -16.92243245, -11.49329874,  -9.12577961,  -8.64231125,
        #     -15.52035586, -10.19347507,  -7.92347709,  -8.68716075, -12.92285367,
        #     -11.53162715, -14.5315762,  -12.39261761, -10.00145136, -15.0961745 ]

        
        # parameters
        TA = forcing1D.TAx + 1j*forcing1D.TAy
        TAx_f,TAy_f = tools.my_fc_filter( dt_forcing, TA, forcing1D.fc)
        TAx = jnp.asarray(TAx_f)
        TAy = jnp.asarray(TAy_f)
        fc = jnp.asarray(forcing1D.fc)
        
        call_args = t0, t1, dt
        
        #mymodel = jslab(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode)
        mymodel = jslab_kt_Ue_Unio(pk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args, use_difx=False)
        var_dfx = inv.Variational(mymodel,observations1D)
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'jslab_kt_Ue_Unio_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)
        
    if TEST_SLAB_KT_2D_ADV:
        print('* test jslab_kt_2D_adv')
        # control vector
        pk = jnp.asarray([-11.31980127, -10.28525189])        
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        
        # parameters
        TAx = jnp.asarray(forcing2D.TAx)
        TAy = jnp.asarray(forcing2D.TAy)
        fc = jnp.asarray(forcing2D.fc)
        Ug = jnp.asarray(forcing2D.Ug)
        Vg = jnp.asarray(forcing2D.Vg)

        dx, dy = 0.1, 0.1
        call_args = t0, t1, dt
        mymodel = jslab_kt_2D_adv(pk, TAx, TAy, Ug, Vg, dx, dy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args,use_difx=False, k_base=k_base)
        var_dfx = inv.Variational(mymodel,observations2D)
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'jslab_kt_2D_adv_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_2D(mymodel, var_dfx, forcing2D, observations2D, name_save, point_loc, LON_bounds, LAT_bounds, path_save_png, dpi)     
    
    if TEST_SLAB_KT_2D_ADV_UT:
        print('* test jslab_kt_2D_adv_Ut')
        
        # control vector
        pk = jnp.asarray([-11.31980127, -10.28525189])     
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        
        # parameters
        TAx = jnp.asarray(forcing2D.TAx)
        TAy = jnp.asarray(forcing2D.TAy)
        fc = jnp.asarray(forcing2D.fc)
        Ug = jnp.asarray(forcing2D.Ug)
        Vg = jnp.asarray(forcing2D.Vg)

        dx, dy = 0.1, 0.1
        call_args = t0, t1, dt
        mymodel = jslab_kt_2D_adv_Ut(pk, TAx, TAy, Ug, Vg, dx, dy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args,use_difx=False, k_base=k_base)
        var_dfx = inv.Variational(mymodel,observations2D)
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'jslab_kt_2D_adv_Ut_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_2D(mymodel, var_dfx, forcing2D, observations2D, name_save, point_loc, LON_bounds, LAT_bounds, path_save_png, dpi)  
        
    if TEST_SLAB_FFT:
        print('* test jslab_fft')
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
                           
        name_save = 'jslab_fft_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)   
      
      
    if TEST_UNSTEAK:
        print('* test junsteak')
        # control vector
        if Nl==1:
            pk = jnp.asarray([-11.31980127, -10.28525189])    
        elif Nl==2:
            pk = jnp.asarray([-10.,-10., -9., -9.])    
        
        
        # parameters
        TAx = jnp.asarray(forcing1D.TAx)
        TAy = jnp.asarray(forcing1D.TAy)
        fc = jnp.asarray(forcing1D.fc)
        
        call_args = t0, t1, dt
        
        #mymodel = jslab(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode)
        mymodel = junsteak(pk, TAx, TAy, fc, dt_forcing, nl=Nl, AD_mode=AD_mode, call_args=call_args, use_difx=False)
        var_dfx = inv.Variational(mymodel,observations1D, filter_at_fc=FILTER_AT_FC)
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'junsteak_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)   
        
        if IDEALIZED_RUN:
            mypk = mymodel.pk
            frc_idealized = forcing.Forcing_idealized_1D(dt_forcing, t0, t1, TAx=0.4, TAy=0., dt_spike=t1-t0)
            step_model = eqx.tree_at(lambda t:t.TAx, mymodel, frc_idealized.TAx)
            step_model = eqx.tree_at(lambda t:t.TAy, step_model, frc_idealized.TAy)
            
            idealized_run(step_model, frc_idealized, name_save, path_save_png, dpi)
        
    if TEST_UNSTEAK_KT:
        print('* test junsteak_kt')
        # control vector
        if Nl==1:
            pk = jnp.asarray([-11.31980127, -10.28525189])    
        elif Nl==2:
            pk = jnp.asarray([-10.,-10., -9., -9.])    
        
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        
        # parameters
        TAx = jnp.asarray(forcing1D.TAx)
        TAy = jnp.asarray(forcing1D.TAy)
        fc = jnp.asarray(forcing1D.fc)
        
        call_args = t0, t1, dt
        
        #mymodel = jslab(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode)
        mymodel = junsteak_kt(pk, TAx, TAy, fc, dTK, dt_forcing, nl=Nl, AD_mode=AD_mode, call_args=call_args, use_difx=False, k_base='gauss')
        var_dfx = inv.Variational(mymodel,observations1D, filter_at_fc=FILTER_AT_FC)
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'junsteak_kt_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi)   
            
            
        if IDEALIZED_RUN:
            mypk = mymodel.pk
            frc_idealized = forcing.Forcing_idealized_1D(dt_forcing, t0, t1, TAx=0.4, TAy=0., dt_spike=dTK)
            step_model = eqx.tree_at(lambda t:t.TAx, mymodel, frc_idealized.TAx)
            step_model = eqx.tree_at(lambda t:t.TAy, step_model, frc_idealized.TAy)
            
            idealized_run(step_model, frc_idealized, name_save, path_save_png, dpi)
       
    if TEST_UNSTEAK_KT_2D:
        print('* test junsteak_kt_2D')
        # control vector
        if Nl==1:
            pk = jnp.asarray([-11.31980127, -10.28525189])    
        elif Nl==2:
            pk = jnp.asarray([-10.,-10., -9., -9.])    
        
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        
        # parameters
        TAx = jnp.asarray(forcing2D.TAx)
        TAy = jnp.asarray(forcing2D.TAy)
        fc = jnp.asarray(forcing2D.fc)
        
        call_args = t0, t1, dt
        
        #mymodel = jslab(pk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode)
        mymodel = junsteak_kt_2D(pk, TAx, TAy, fc, dTK, dt_forcing, nl=Nl, AD_mode=AD_mode, call_args=call_args, use_difx=False, k_base='gauss')
        var_dfx = inv.Variational(mymodel,observations2D, filter_at_fc=FILTER_AT_FC)
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx)   

        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
                           
        name_save = 'junsteak_kt_2D_'+namesave_loc
        if PLOT_TRAJ:
            plot_traj_2D(mymodel, var_dfx, forcing2D, observations2D, name_save, point_loc, LON_bounds, LAT_bounds, path_save_png, dpi)   
            
            
        if IDEALIZED_RUN:
            mypk = mymodel.pk
            frc_idealized = forcing.Forcing_idealized_2D(dt_forcing, t0, t1, TAx=0.4, TAy=0., dt_spike=dTK)
            step_model = eqx.tree_at(lambda t:t.TAx, mymodel, frc_idealized.TAx)
            step_model = eqx.tree_at(lambda t:t.TAy, step_model, frc_idealized.TAy)
            
            idealized_run(step_model, frc_idealized, name_save, path_save_png, dpi)
            
    end = clock.time()
    print('Total execution time = '+str(jnp.round(end-start,2))+' s')
    plt.show()