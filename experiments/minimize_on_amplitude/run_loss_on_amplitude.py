"""
This script is using a loss function on the amplitude of the complex current.
Goal: is the reconstructed current improved compared to a minimization process with comparison of raw trajectories ?




Note: the result of the minimization depends on the starting point. For 2 parameters only models, we can plot the map of J
        to choose a point 'far but not too far' from the solution. For models with more parameters this cannot be done.
"""




import numpy as np
import time as clock
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('qtagg')
import sys
import os
import xarray as xr
sys.path.insert(0, '../../src')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
sys.path.insert(0, '../../tests_models')
import jax
import jax.numpy as jnp
import equinox as eqx
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
# model parameters
Nl                  = 2            # number of layers for multilayer models
dTK                 = 10*oneday     # how much vectork K changes with time, basis change to 'k_base'      
extra_args = {'AD_mode':'F',        # forward mode for AD (for diffrax' diffeqsolve)
            'use_difx':False,       # use diffrax solver to time integrate
            'k_base':'gauss'}       # base of K transform. 'gauss' or 'id'    


ON_PAPA             = False      # use PAPA station data, only for 1D models
ON_1D               = True      # use CROCO data at 'point_loc'
ON_2D               = False

# run parameters
t0                  = 60*oneday    # start day 
t1                  = 100*oneday    # end day
t0_papa             = 20*oneday
t1_papa             = 50*oneday

dt                  = 60.           # timestep of the model (s) 

# What to test
MINIMIZE            = False      # Does the model converges to a solution ?
maxiter             = 50        # if MINIMIZE: max number of iteration
SAVE_AS_NC          = True      # for 2D models


PLOT                = True
PLOT_TRAJ               = True      # Show a trajectory
PLOT_SNAPSHOT           = True      # 2D models: XY snapshot

# what to test
L_model_to_test_1D    = ['jslab','junsteak','jslab_kt','junsteak_kt'] 
L_model_to_test_2D    = ['jslab_kt_2D','junsteak_kt_2D']


# PLOT
dpi=200
path_save_png = './pngs/'
t0_plot                  = 60*oneday    # start day 
t1_plot                  = 100*oneday    # end day
t0_papa_plot             = 25*oneday
t1_papa_plot             = 35*oneday

path_saved_models = './saved_models/'
path_save_output = './saved_outputs/'


# =================================
# Forcing, OSSE and observations
# =================================
# 1D
point_loc = [-49.,39.] # march 8th 0300, t0=60 t1=100, centered on an eddy

# 2D
R = 5. # 5.0 
LON_bounds = [point_loc[0]-R,point_loc[0]+R]
LAT_bounds = [point_loc[1]-R,point_loc[1]+R]
# Forcing
dt_forcing          = onehour      # forcing timestep

# OSSE from Croco
dt_OSSE             = onehour      # timestep of the OSSE (s)
path_regrid = '../../data_regrid/'
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

path_papa = '../../data_PAPA_2018/'
name_papa = ['cur50n145w_hr.nc',
              'd50n145w_hr.nc',
              'lw50n145w_hr.nc',
              'rad50n145w_hr.nc',
              's50n145w_hr.nc',
              'sss50n145w_hr.nc',
              'sst50n145w_hr.nc',
              't50n145w_hr.nc',
              'w50n145w_hr.nc',
              'tau50n145w_hr.nc']

# Observations
period_obs          = oneday #86400      # s, how many second between observations  

# ============================================================
# END PARAMETERS
# ============================================================
os.system('mkdir -p '+path_save_png)
os.system('mkdir -p '+path_saved_models)
os.system('mkdir -p '+path_save_output)
namesave_loc = str(point_loc[0])+'E_'+str(point_loc[1])+'N'
namesave_loc_area = str(point_loc[0])+'E_'+str(point_loc[1])+'N_R'+str(R)


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
    
    
     # forcing
    if ON_PAPA:
        PAPA_forcing = forcing.Forcing_from_PAPA(dt_forcing, t0_papa, t1_papa, file_papa)
        PAPA_observation = observations.Observation_from_PAPA(period_obs, t0_papa, t1_papa, dt_forcing, file_papa)
    if ON_1D:
        CROCO_1D_forcing = forcing.Forcing1D(point_loc, t0, t1, dt_forcing, file)
        CROCO_1D_obs    = observations.Observation1D(point_loc, period_obs, t0, t1, dt_OSSE, file)
    
    # CROCO_2D_forcing = 
    # CROCO_2D_obs    = 
    
    
    dict_model = {} 
    dict_model['PAPA'] = {}
    dict_model['1D'] = {}
    dict_model['2D'] = {}
    for USE_AMPLITUDE in [False, True]:
     
        if USE_AMPLITUDE:
            txt_add_amplitude = 'amp' # amplitude
        else:
            txt_add_amplitude = 'cur' # current
        
        
        
        ######################
        # ON PAPA
        ######################
        if ON_PAPA:
            print('**************')
            print(' Location is PAPA station')
            print(f' if multi layer, nl={Nl}')
            print(f' use_amplitude = {USE_AMPLITUDE}')
            print('**************\n')
            dict_model['PAPA'][txt_add_amplitude] = {}
            
            txt_add_location = f'{txt_add_amplitude}_t{int(t0_papa/oneday)}_t{int(t1_papa/oneday)}'
            os.system('mkdir -p '+path_saved_models+'PAPA/')
            
            for model_name in L_model_to_test_1D:
                print('* test '+model_name)
                name_save = model_name+'_PAPA'
                
                
                    
                call_args = t0_papa, t1_papa, dt
                args_model = {'dTK':dTK, 'Nl':Nl}
                args_2D = {}
                
                # model initialization
                mymodel = model_instanciation(model_name, PAPA_forcing, args_model, call_args, extra_args)
                var_dfx = inv.Variational(mymodel, PAPA_observation, use_amplitude=USE_AMPLITUDE)
                if MINIMIZE:
                    print(' minimizing ...')
                    t7 = clock.time()
                    mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
                    print(' time, minimize',clock.time()-t7)
                    # save the model
                    eqx.tree_serialise_leaves(path_saved_models+f'PAPA/best_{model_name}_{txt_add_location}.pt', mymodel)
                   
                # get back the models
                dict_model['PAPA'][txt_add_amplitude][model_name] = eqx.tree_deserialise_leaves(path_saved_models+f'PAPA/best_{model_name}_{txt_add_location}.pt', mymodel)
            
        
        
        ######################
        # 1D Croco
        ######################    
        if ON_1D:
            indx = tools.nearest(dsfull.lon.values,point_loc[0])
            indy = tools.nearest(dsfull.lat.values,point_loc[1])
            txt_location = f'{np.round(dsfull.lon.values[indx],2)}°E, {np.round(dsfull.lat.values[indy],2)}°N'
            print('**************')
            print(' Location is '+txt_location)
            print(f' if multi layer, nl={Nl}')
            print(f' use_amplitude = {USE_AMPLITUDE}')
            print('**************\n')
        
            dict_model['1D'][txt_add_amplitude] = {}
            
            txt_add_location = f'{txt_add_amplitude}_t{int(t0/oneday)}_t{int(t1/oneday)}'
            os.system('mkdir -p '+path_saved_models+'1D/')
            
            for model_name in L_model_to_test_1D:
                print('* test '+model_name)
                name_save = model_name+'_1D'
                
                
                    
                call_args = t0, t1, dt
                args_model = {'dTK':dTK, 'Nl':Nl}
                args_2D = {}
                
                # model initialization
                mymodel = model_instanciation(model_name, CROCO_1D_forcing, args_model, call_args, extra_args)
                var_dfx = inv.Variational(mymodel, CROCO_1D_obs, use_amplitude=USE_AMPLITUDE)
                if MINIMIZE:
                    print(' minimizing ...')
                    t7 = clock.time()
                    mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
                    print(' time, minimize',clock.time()-t7)
                    # save the model
                    eqx.tree_serialise_leaves(path_saved_models+f'1D/best_{model_name}_{txt_add_location}.pt', mymodel)
                   
                # get back the models
                dict_model['1D'][txt_add_amplitude][model_name] = eqx.tree_deserialise_leaves(path_saved_models+f'1D/best_{model_name}_{txt_add_location}.pt', mymodel)
        
        
        
        ######################
        # 2D Croco
        ######################  
        if ON_2D:
            indx = tools.nearest(dsfull.lon.values,point_loc[0])
            indy = tools.nearest(dsfull.lat.values,point_loc[1])
            txt_location = f'{dsfull.lon.values[indx]}°E, {dsfull.lat.values[indy]}°N'
            print('**************')
            print(' Location is '+txt_location)
            print(f' 2D slice is {LON_bounds}°E {LAT_bounds}°N')
            print(f' if multi layer, nl={Nl}')
            print(f' use_amplitude = {USE_AMPLITUDE}')
            print('**************\n')
    

    if PLOT:
        """"""
        dir = 0
        
        # at PAPA
        if ON_PAPA:
            os.system(f'mkdir -p {path_save_png}PAPA/')
            if PLOT_TRAJ:
                
                truth = PAPA_forcing.U, PAPA_forcing.V
                xtime = (t0_papa + PAPA_forcing.time)/oneday
                xtime_obs = (t0_papa + PAPA_observation.time_obs)/oneday
                obs = PAPA_observation.get_obs()[dir]
                fc = PAPA_forcing.fc
                
                truth_nio = tools.my_fc_filter(dt_forcing, truth[0]+1j*truth[1], fc)
                
                for model_name in dict_model['PAPA']['amp']:
                    traj_amp = dict_model['PAPA']['amp'][model_name](save_traj_at=mymodel.dt_forcing)
                    traj_cur = dict_model['PAPA']['cur'][model_name](save_traj_at=mymodel.dt_forcing)
                    if model_name in L_nlayers_models:
                        traj_amp = traj_amp[0][:,0], traj_amp[0][:,1] # <- get first layer currents
                        traj_cur = traj_cur[0][:,0], traj_cur[0][:,1]
                        
                    # compute rmse
                    myRMSE_amp = tools.score_RMSE(traj_amp, truth)    
                    myRMSE_cur = tools.score_RMSE(traj_cur, truth)
                    
                    fig, ax = plt.subplots(2,1,figsize = (10,8),constrained_layout=True,dpi=dpi)
                    ax[0].plot(xtime, traj_amp[dir], label=model_name+f' amp ({np.round(myRMSE_amp*100,2)})',c='b')
                    ax[0].plot(xtime, traj_cur[dir], label=model_name+f' cur ({np.round(myRMSE_cur*100,2)})',c='c')
                    ax[0].plot(xtime, truth[dir], label='truth', c='k', alpha=1)
                    # ax[0].plot(xtime, truth_nio[dir], label='truth_nio',c='k')
                    ax[0].scatter(xtime_obs, obs, label='obs', marker='o', c='r')
                    ax[0].legend(loc='lower left')
                    ax[0].set_ylabel('ageo current (m/s)')
                    ax[0].set_ylim([-0.4,0.4])
                    ax[0].grid()
                    ax[0].set_xlim([t0_papa_plot/oneday, t1_papa_plot/oneday])
                    ax[1].plot(xtime, PAPA_forcing.TAx, c='g', label=r'$\tau_x$')
                    ax[1].plot(xtime, PAPA_forcing.TAy, c='orange', label=r'$\tau_y$')
                    ax[1].set_ylabel('wind stress (N/m2)')
                    ax[1].set_xlabel('time (days)')
                    ax[1].set_xlim([t0_papa_plot/oneday, t1_papa_plot/oneday])
                    ax[1].set_ylim([-1.2,1.2])
                    ax[1].grid()
                    ax[1].legend(loc='lower left')
                    fig.savefig(path_save_png+f'PAPA/{model_name}_t{int(t0_papa_plot/oneday)}_t{int(t1_papa_plot/oneday)}.png')
                    
            
        if ON_1D:
            os.system(f'mkdir -p {path_save_png}CROCO_1D/')
            if PLOT_TRAJ:
                
                truth = CROCO_1D_forcing.U, CROCO_1D_forcing.V
                xtime = (t0 + CROCO_1D_forcing.time)/oneday
                xtime_obs = (t0 + CROCO_1D_obs.time_obs)/oneday
                obs = CROCO_1D_obs.get_obs()[dir]
                fc = CROCO_1D_obs.fc
                
                truth_nio = tools.my_fc_filter(dt_forcing, truth[0]+1j*truth[1], fc)
                
                for model_name in dict_model['1D']['amp']:
                    traj_amp = dict_model['1D']['amp'][model_name](save_traj_at=mymodel.dt_forcing)
                    traj_cur = dict_model['1D']['cur'][model_name](save_traj_at=mymodel.dt_forcing)
                    if model_name in L_nlayers_models:
                        traj_amp = traj_amp[0][:,0], traj_amp[0][:,1] # <- get first layer currents
                        traj_cur = traj_cur[0][:,0], traj_cur[0][:,1]
                        
                    # compute rmse
                    myRMSE_amp = tools.score_RMSE(traj_amp, truth)    
                    myRMSE_cur = tools.score_RMSE(traj_cur, truth)
                    
                    fig, ax = plt.subplots(2,1,figsize = (10,8),constrained_layout=True,dpi=dpi)
                    ax[0].plot(xtime, traj_amp[dir], label=model_name+f' amp ({np.round(myRMSE_amp*100,2)})',c='b')
                    ax[0].plot(xtime, traj_cur[dir], label=model_name+f' cur ({np.round(myRMSE_cur*100,2)})',c='c')
                    ax[0].plot(xtime, truth[dir], label='truth', c='k', alpha=0.5)
                    ax[0].plot(xtime, truth_nio[dir], label='truth_nio',c='k')
                    ax[0].scatter(xtime_obs, obs, label='obs', marker='o', c='r')
                    ax[0].legend(loc='lower left')
                    ax[0].set_ylabel('ageo current (m/s)')
                    ax[0].set_ylim([-0.6,0.6])
                    ax[0].grid()
                    ax[0].set_xlim([t0_plot/oneday, t1_plot/oneday])
                    ax[1].plot(xtime, CROCO_1D_forcing.TAx, c='g', label=r'$\tau_x$')
                    ax[1].plot(xtime, CROCO_1D_forcing.TAy, c='orange', label=r'$\tau_y$')
                    ax[1].set_ylabel('wind stress (N/m2)')
                    ax[1].set_xlabel('time (days)')
                    ax[1].set_xlim([t0_plot/oneday, t1_plot/oneday])
                    ax[1].set_ylim([-3,3])
                    ax[1].grid()
                    ax[1].legend(loc='lower left')
                    fig.savefig(path_save_png+f'CROCO_1D/{model_name}_t{int(t0_plot/oneday)}_t{int(t1_plot/oneday)}.png')
                
        if ON_2D:
            os.system(f'mkdir -p {path_save_png}CROCO_2D/')
            if PLOT_TRAJ:
                """"""
            if PLOT_SNAPSHOT:
                """"""  
                
                
    plt.show()
    
    
    
    """
    TO DO list:
    
        - PAPA plot
        
            -> add RMSE of each models reconstruction in the legend
            
        - other
        
            croco 1D: add minimization process and plot
            croco 2D: add minimization process and plot
    
    
    
    """
