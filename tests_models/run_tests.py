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

# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_triton_gemm=true "
#     "--xla_gpu_enable_latency_hiding_scheduler=true "
#     "--xla_gpu_enable_highest_priority_async_stream=true "
# )

import jax
import jax.numpy as jnp
import equinox as eqx
#import jax
# jax.config.update('jax_platform_name', 'cpu')
#jax.config.update("jax_transfer_guard", "log_explicit") 

# my inmports
from models import classic_slab, unsteak
from basis import kt_ini

import forcing
import inv
import observations
from tests_functions import *
import tools
from constants import *
from Listes_models import *

start = clock.time()

# ============================================================
# PARAMETERS
# ============================================================
#ON_HPC      = False      # on HPC

# model parameters
Nl                  = 2            # number of layers for multilayer models
dTK                 = 10*oneday     # how much vectork K changes with time, basis change to 'k_base'      
extra_args = {'AD_mode':'F',        # forward mode for AD (for diffrax' diffeqsolve)
            'use_difx':False,       # use diffrax solver to time integrate
            'k_base':'gauss'}       # base of K transform. 'gauss' or 'id'

# run parameters
t0                  = 60*oneday    # start day 
t1                  = 100*oneday    # end day
dt                  = 60.           # timestep of the model (s) 

# What to test
PLOT_TRAJ           = True      # Show a trajectory
FORWARD_PASS        = False      # How fast the model is running ?
MINIMIZE            = True      # Does the model converges to a solution ?
maxiter             = 50        # if MINIMIZE: max number of iteration
MAKE_FILM           = False      # for 2D models, plot each hour
SAVE_AS_NC          = True      # for 2D models
MEM_PROFILER        = False       # memory profiler


ON_PAPA             = False      # use PAPA station data, only for 1D models
FILTER_AT_FC        = False      # minimize filtered ageo current with obs if model has this option
IDEALIZED_RUN       = False       # try the model on a step wind stress

# Switches
L_model_to_test             = ['junsteak_kt_2D'] #'junsteak_kt_2D_adv'] # L_all

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
point_loc = [-49.,39.] # march 8th 0300, t0=60 t1=100, centered on an eddy

# 2D
R = 5. # 5.0 
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
    
    for model_name in L_model_to_test:
        print('* test '+model_name)
        name_save = model_name+'_'+namesave_loc
        
        # forcing
        if model_name in L_1D_models:
            if ON_PAPA:
                myforcing = forcing.Forcing_from_PAPA(dt_forcing, t0, t1, file_papa)
                myobservation = observations.Observation_from_PAPA(period_obs, t0, t1, dt_forcing, file_papa)
            else:
                myforcing = forcing.Forcing1D(point_loc, t0, t1, dt_forcing, file)
                myobservation = observations.Observation1D(point_loc, period_obs, t0, t1, dt_OSSE, file)
        elif model_name in L_2D_models:
            myforcing = forcing.Forcing2D(dt_forcing, t0, t1, file, LON_bounds, LAT_bounds)
            myobservation = observations.Observation2D(period_obs, t0, t1, dt_OSSE, file, LON_bounds, LAT_bounds)
        
        if model_name in L_unsteaks:
            module_name = 'unsteak'
        elif model_name in L_slabs:
            module_name = 'classic_slab'
        
        """
        TO DO:        
        - clean up tests_functions module.
        - clean up models modules.
        """      
     
        call_args = t0, t1, dt
        args_model = {'dTK':dTK, 'Nl':Nl}
        args_2D = {}
        
        # model initialization
        #classname = getattr(sys.modules[module_name], model_name)
        mymodel = model_instanciation(model_name, myforcing, args_model, call_args, extra_args)
        var_dfx = inv.Variational(mymodel, myobservation, filter_at_fc=FILTER_AT_FC)
        
        if FORWARD_PASS:
            run_forward_cost_grad(mymodel, var_dfx) 
       
        if MINIMIZE:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time, minimize',clock.time()-t7)
            # save the model
            eqx.tree_serialise_leaves('./saved_outputs/'+f'best_{model_name}.pt', mymodel)
        else:
            # open the saved model
            mymodel = eqx.tree_deserialise_leaves('./saved_outputs/'+f'best_{model_name}.pt', mymodel)

        
        if PLOT_TRAJ:
            if model_name in L_1D_models:
                plot_traj_1D(mymodel, call_args, var_dfx, myforcing, myobservation, name_save, path_save_png, dpi)
            elif model_name in L_2D_models:
                attime = 70 # in days
                plot_traj_2D(mymodel, var_dfx, myforcing, myobservation, name_save, point_loc, LON_bounds, LAT_bounds, attime=attime, path_save_png=path_save_png, dpi=dpi)

        if MEM_PROFILER:
            memory_profiler(mymodel)
            
        ###############################################
        # only 2D
        if model_name in L_2D_models:
            if IDEALIZED_RUN:
                mypk = mymodel.pk
                frc_idealized = forcing.Forcing_idealized_2D(dt_forcing, t0, t1, TAx=0.4, TAy=0., dt_spike=dTK)
                step_model = eqx.tree_at(lambda t:t.TAx, mymodel, frc_idealized.TAx)
                step_model = eqx.tree_at(lambda t:t.TAy, step_model, frc_idealized.TAy)
                
                idealized_run(step_model, frc_idealized, name_save, path_save_png, dpi)

            if MAKE_FILM:
                make_film(mymodel, myforcing, LON_bounds, LAT_bounds, namesave_loc_area, path_save_png)
                
            if SAVE_AS_NC:
                save_output_as_nc(mymodel, myforcing, LON_bounds, LAT_bounds, name_save, where_to_save='./saved_outputs/')
        ###############################################   
            
            
            
            
    end = clock.time()
    print('Total execution time = '+str(jnp.round(end-start,2))+' s')
    plt.show()