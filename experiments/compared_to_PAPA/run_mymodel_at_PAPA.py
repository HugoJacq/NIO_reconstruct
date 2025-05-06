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
jax.config.update('jax_platform_name', 'cpu')


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

L_model_to_test     = ['jslab','jslab_kt','junsteak','junsteak_kt'] #   'jslab',

# model parameters
Nl                  = 2                         # number of layers for multilayer models
dTK                 = 10*oneday                 # how much vectork K changes with time, basis change to 'k_base'      
extra_args          = {'AD_mode':'F',           # forward mode for AD (for diffrax' diffeqsolve)
                        'use_difx':False,       # use diffrax solver to time integrate
                        'k_base':'gauss'}       # base of K transform. 'gauss' or 'id'
FILTER_AT_FC        = False


# run parameters
t0                  = 20*oneday
t1                  = 50*oneday
dt                  = 60.        # timestep of the model (s) 

# What to do
SAVE_PKs            = False         # minimzize and save PKs
maxiter             = 100           # max number of iteration for minimization
PLOT                = True          # plot or not
NORM_FREQ           = False         # normalise frequency in PSD plots
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
              'w50n145w_hr.nc',
              'tau50n145w_hr.nc']

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
        print('')
        print('* Working on '+model_name)
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
            fc = mymodel.fc
            
            # estimate of current 
            sol = mymodel()
            if model_name in L_nlayers_models:
                Ua,Va = sol[0][:,0], sol[1][:,0]
            else:
                Ua,Va = sol[0], sol[1]
            
            # truth
            U,V = myforcing.U,myforcing.V
            U_nio,V_nio = tools.my_fc_filter(mymodel.dt_forcing,U+1j*V, fc)
            
            # observations
            step_obs = int(period_obs)//int(dt_forcing)
            Uobs,Vobs = myobservation.get_obs()
            timeobs = myobservation.time_obs
            
            # RMSE
            skip = 3
            sRMSE = tools.score_RMSE((Ua[skip:],Va[skip:]), (U[skip:],V[skip:]))                # <- RMSE w.r.t. the full U
            sRMSE_nio = tools.score_RMSE((Ua[skip:],Va[skip:]), (U_nio[skip:],V_nio[skip:])) 
            # sRMSE = tools.score_RMSE((Ua[::step_obs],Va[::step_obs]), (Uobs,Vobs))    # <- RMSE w.r.t. only observations
            
            fig, ax = plt.subplots(2,1,figsize = (6,5),constrained_layout=True,dpi=dpi)
            ax[0].set_title(model_name+f' at PAPA\nRMSE={np.round(sRMSE,5)} RMSE_nio={np.round(sRMSE_nio,5)}')
            ax[0].plot(myforcing.time/oneday, U, label='truth', c='k', alpha=0.5)
            ax[0].plot(myforcing.time/oneday, U_nio, label='truth NIO', c='k', alpha=1)
            ax[0].plot(myforcing.time/oneday, Ua, label=model_name, c='b')
            ax[0].scatter(timeobs/oneday, Uobs, c='r', marker='x')
            ax[0].set_ylabel('U (m/s)')
            ax[0].set_ylim([-0.5,0.5])
            ax[0].legend()
            
            ax[1].plot(myforcing.time/oneday, myforcing.TAx, label=r'$\tau_x$', c='b')
            ax[1].plot(myforcing.time/oneday, myforcing.TAy, label=r'$\tau_y$', c='orange')
            ax[1].set_ylabel('wind stress N/m2')
            ax[1].legend()
            ax[1].set_ylim([-1.5,1.5])
            ax[1].set_xlabel('days')
            for axe in ax:
                axe.grid()
            fig.savefig(f'{path_save_png}zonal_current_{model_name}.png')
            
            
            # PSD and PSD score
            if False:
                # Rotary spectra
                ff, CWr, ACWr, CWe, ACWe = tools.rotary_spectra(1., Ua, Va, myforcing.U, myforcing.V)
                wsize = 21
                CWe_smoothed = tools.savitzky_golay(CWe, window_size=wsize, order=4, deriv=0, rate=1)
                CWr_smoothed = tools.savitzky_golay(ACWe, window_size=wsize, order=4, deriv=0, rate=1)
            
                if NORM_FREQ:
                    mean_fc = 2*2*np.pi/86164*np.sin(point_loc[1]*np.pi/180)*onehour/(2*np.pi)
                    xtxt = r'f/$f_c$'
                else:
                    mean_fc = 1
                    xtxt = 'h-1'
                fig, axs = plt.subplots(2,1,figsize=(7,6), gridspec_kw={'height_ratios': [4, 1.5]})
                axs[0].loglog(ff/mean_fc,CWr_smoothed, c='k', label='reference')
                axs[0].loglog(ff/mean_fc,CWe_smoothed, c='b', label='error (model - truth)')
                #axs[0].axis([2e-3,2e-1, 1e-4,2e0])
                axs[0].grid('on', which='both')
                axs[1].set_xlabel(xtxt)
                axs[0].legend()
                axs[0].set_ylabel('Clockwise PSD (m2/s2)')
                axs[0].title.set_text(model_name +f', RMSE={np.round(sRMSE,5)}')

                axs[1].semilogx(ff/mean_fc,(1-CWe_smoothed/CWr_smoothed)*100, c='b', label='Reconstruction Score')
                #axs[1].axis([2e-3,2e-1,0,100])
                axs[1].set_ylim([0,100])
                axs[1].grid('on', which='both')
                axs[1].set_ylabel('Scores (%)')
                fig.savefig(f'{path_save_png}PSD_score_{model_name}.png')
            
            
    plt.show()