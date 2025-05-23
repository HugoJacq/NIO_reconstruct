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


ON_PAPA             = True      # use PAPA station data, only for 1D models
ON_1D               = False
ON_2D               = False

# run parameters
t0                  = 60*oneday    # start day 
t1                  = 100*oneday    # end day
t0_papa             = 60*oneday
t1_papa             = 100*oneday


dt                  = 60.           # timestep of the model (s) 

# What to test
MINIMIZE            = False      # Does the model converges to a solution ?
maxiter             = 50        # if MINIMIZE: max number of iteration
SAVE_AS_NC          = True      # for 2D models


PLOT = True
PLOT_TRAJ           = True      # Show a trajectory
PLOT_SNAPSHOT       = True

# what to test
L_model_to_test_1D    = ['jslab','junsteak','jslab_kt','junsteak_kt'] 
L_model_to_test_2D    = ['jslab_kt_2D','junsteak_kt_2D']


# PLOT
dpi=200
path_save_png = './pngs/'


path_saved_models = './saved_models/'
path_save_output = './saved_outputs/'


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
    PAPA_forcing = forcing.Forcing_from_PAPA(dt_forcing, t0_papa, t1_papa, file_papa)
    PAPA_observation = observations.Observation_from_PAPA(period_obs, t0_papa, t1_papa, dt_forcing, file_papa)
    # CROCO_1D_forcing =
    # CROCO_1D_obs    =
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
            print('Location is PAPA station')
            print(f'if multi layer, nl={Nl}')
            print('**************\n')
            
            
            dict_model['PAPA'][txt_add_amplitude] = {}
            
            txt_add_location = f'{txt_add_amplitude}_t{int(t0_papa/oneday)}_t{int(t1_papa/oneday)}'
            os.system('mkdir -p '+path_saved_models+'./')
            
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
                
                    
                dict_model['PAPA'][txt_add_amplitude][model_name] = eqx.tree_deserialise_leaves(path_saved_models+f'PAPA/best_{model_name}_{txt_add_location}.pt', mymodel)
            
        ######################
        # 1D Croco
        ######################    
        if ON_1D:
            indx = tools.nearest(dsfull.lon.values,point_loc[0])
            indy = tools.nearest(dsfull.lat.values,point_loc[1])
            txt_location = f'{dsfull.lon.values[indx]}°E, {dsfull.lat.values[indy]}°N'
            print('**************')
            print('Location is '+txt_location)
            print(f'if multi layer, nl={Nl}')
            print('**************\n')
        
        
        
        
        
        ######################
        # 2D Croco
        ######################  
        if ON_2D:
            indx = tools.nearest(dsfull.lon.values,point_loc[0])
            indy = tools.nearest(dsfull.lat.values,point_loc[1])
            txt_location = f'{dsfull.lon.values[indx]}°E, {dsfull.lat.values[indy]}°N'
            print('**************')
            print('Location is '+txt_location)
            print(f'2D slice is {LON_bounds}°E {LAT_bounds}°N')
            print(f'if multi layer, nl={Nl}')
            print('**************\n')
    

    if PLOT:
        """"""
        dir = 0
        
        # at PAPA
        if ON_PAPA:
            
            if PLOT_TRAJ:
                
                truth = PAPA_forcing.U, PAPA_forcing.V
                xtime = PAPA_forcing.time
                fc = PAPA_forcing.fc
                
                truth_nio = tools.my_fc_filter(dt_forcing, truth[0]+1j*truth[1], fc)
                
                
                print(dict_model['PAPA'].keys())
                
                for model_name in dict_model['PAPA']['amp']:
                    traj_amp = dict_model['PAPA']['amp'][model_name](save_traj_at=mymodel.dt_forcing)
                    traj_cur = dict_model['PAPA']['cur'][model_name](save_traj_at=mymodel.dt_forcing)
                    
                    fig, ax = plt.subplots(2,1,figsize = (10,8),constrained_layout=True,dpi=dpi)
                    ax[0].plot(xtime, traj_amp[dir], label=model_name+' amp',c='b')
                    ax[0].plot(xtime, traj_cur[dir], label=model_name+' cur',c='c')
                    ax[0].plot(xtime, truth[dir], label='truth', c='k', alpha=0.5)
                    ax[0].plot(xtime, truth_nio[dir], label='truth_nio',c='k')
                    
                    ax[0].legend()
            
            if PLOT_SNAPSHOT:
                """"""  
    plt.show()
