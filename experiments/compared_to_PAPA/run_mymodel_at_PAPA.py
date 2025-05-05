"""
Here we want to have a look on models performance at PAPA station
"""


import numpy as np
import time as clock
import matplotlib.pyplot as plt
import sys
import os
import xarray as xr
import equinox as eqx

sys.path.insert(0, '../../src')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
import jax
import jax.numpy as jnp

from models.classic_slab import jslab, jslab_kt, jslab_kt_2D
from basis import kt_ini, kt_1D_to_2D, pkt2Kt_matrix
import forcing
import inv
import observations
import tools
from constants import *

sys.path.insert(0, '../../tests_models')
from tests_functions import *

start = clock.time()


# ============================================================
# PARAMETERS
# ============================================================

L_model_to_test     = ['jslab','junsteak_kt'] #  

# model parameters
Nl                  = 2                         # number of layers for multilayer models
dTK                 = 10*oneday                 # how much vectork K changes with time, basis change to 'k_base'      
extra_args          = {'AD_mode':'F',           # forward mode for AD (for diffrax' diffeqsolve)
                        'use_difx':False,       # use diffrax solver to time integrate
                        'k_base':'gauss'}       # base of K transform. 'gauss' or 'id'
FILTER_AT_FC        = False


# run parameters
t0                  = 0*oneday
t1                  = 50*oneday
dt                  = 60.        # timestep of the model (s) 

# What to do
SAVE_PKs            = False
maxiter             = 100         # max number of iteration for minimization
PLOT                = True

# PLOT
dpi=200


path_save_models    = './saved_models/'
path_save_png       = './png_models_at_PAPA/'

# =================================
# Forcing, OSSE and observations
# =================================

# PAPA station is located at 50.1°N, 144.9°W
point_loc = [-144.9, 50.1]

# Forcing : PAPA data is hourly 
dt_forcing  = onehour      # forcing timestep

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
os.system('mkdir -p '+path_save_models)

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
    
    
    myforcing = forcing.Forcing_from_PAPA(dt_forcing, t0, t1, file)
    myobservation = observations.Observation_from_PAPA(period_obs, t0, t1, dt_forcing, file)
    
    
    call_args = t0, t1, dt
    args_model = {'dTK':dTK, 'Nl':Nl}
    
    for model_name in L_model_to_test:
        mymodel = model_instanciation(model_name, myforcing, args_model, call_args, extra_args)
        var_dfx = inv.Variational(mymodel, myobservation, filter_at_fc=FILTER_AT_FC)    


        if SAVE_PKs:
            print(' minimizing ...')
            t7 = clock.time()
            mymodel, _ = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            print(' time minimization:',clock.time()-t7)
            eqx.tree_serialise_leaves(path_save_models+model_name+'.pt',mymodel)
            
            
        if PLOT:
            
            mymodel = eqx.tree_deserialise_leaves(path_save_models+model_name+'.pt',   # <- getting the saved PyTree 
                                                mymodel                    # <- here the call is just to get the structure
                                                )
            # estimate of current 
            sol = mymodel()
            if model_name in L_nlayers_models:
                Ua,Va = sol[0][:,0], sol[1][:,0]
            else:
                Ua,Va = sol[0], sol[1]
            
            # observations
            step_obs = int(period_obs)//int(dt_forcing)
            Uobs,Vobs = myobservation.get_obs()
            timeobs = myobservation.time_obs
            
            # RMSE
            
            # sRMSE = tools.score_RMSE((Ua[::step_obs],Va[::step_obs]), (Uobs,Vobs))
            sRMSE = tools.score_RMSE((Ua,Va), (myforcing.U,myforcing.V))
            
            fig, ax = plt.subplots(2,1,figsize = (10,10),constrained_layout=True,dpi=dpi)
            ax[0].set_title(model_name+f' at PAPA, t0={t0/oneday} t1={t1/oneday} days, RMSE={np.round(sRMSE,4)}')
            ax[0].plot(myforcing.time/oneday, myforcing.U, label='truth', c='k')
            ax[0].plot(myforcing.time/oneday, Ua, label=model_name, c='b')
            ax[0].scatter(timeobs/oneday, Uobs, c='r', marker='x')
            ax[1].set_xlabel('days')
            ax[0].set_ylabel('U (m/s)')
            ax[0].set_ylim([-0.5,0.5])
            ax[0].legend()
            ax[1].plot(myforcing.time/oneday, myforcing.TAx, label=r'$\tau_x$', c='b')
            ax[1].plot(myforcing.time/oneday, myforcing.TAy, label=r'$\tau_y$', c='orange')
            ax[1].set_ylabel('wind stress as Cd*U**2')
            ax[1].legend()
            fig.savefig(f'{path_save_png}zonal_current_{model_name}.png')
            
            
            
    plt.show()