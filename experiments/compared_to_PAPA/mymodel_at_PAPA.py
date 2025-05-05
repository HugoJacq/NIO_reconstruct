"""
Here we want to have a look on models performance at PAPA station
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

from models.classic_slab import jslab, jslab_kt, jslab_kt_2D
from basis import kt_ini, kt_1D_to_2D, pkt2Kt_matrix
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

# model parameters
Nl                  = 1         # number of layers for multilayer models
dTK                 = 10*oneday   # how much vectork K changes with time, basis change to exp
k_base              = 'gauss'   # base of K transform. 'gauss' or 'id'
AD_mode             = 'F'       # forward mode for AD 

# run parameters
t0                  = 25*oneday
t1                  = 50*oneday
dt                  = 60.        # timestep of the model (s) 

# What to test
FORWARD_PASS        = False      # tests forward, cost, gradcost
MINIMIZE            = True      # switch to do the minimisation process
maxiter             = 100         # max number of iteration
PLOT_TRAJ           = True


TEST_SLAB                   = False
TEST_SLAB_KT                = True

# PLOT
dpi=200
path_save_png = './png_models_at_PAPA/'

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
    
    
    forcing1D = forcing.Forcing_from_PAPA(dt_forcing, t0, t1, file)
    observations1D = observations.Observation_from_PAPA(period_obs, t0, t1, dt_forcing, file)
    
    print('My slab at station PAPA')
    # control vector
    pk = jnp.asarray([-11.31980127, -10.28525189])   
        
    # parameters
    TAx = jnp.asarray(forcing1D.TAx)
    TAy = jnp.asarray(forcing1D.TAy)
    fc = jnp.asarray(forcing1D.fc)
    
    call_args = t0, t1, dt
    
    # Model definition
    if TEST_SLAB_KT:
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        mymodel = jslab_kt(pk, TAx, TAy, fc, dTK, dt_forcing, AD_mode=AD_mode, call_args=call_args, k_base=k_base)
    elif TEST_SLAB:
        mymodel = jslab(pk, TAx, TAy, fc, dt_forcing, AD_mode=AD_mode, call_args=call_args)
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
        Ua,Va = mymodel(save_traj_at=mymodel.dt_forcing)
        U = forcing1D.data.U.values
        V = forcing1D.data.V.values
        t0 = mymodel.t0
        Uo, _ = observations1D.get_obs()
        RMSE = tools.score_RMSE(Ua, U) 
        dynamic_model, static_model = my_partition(mymodel)
        final_cost = var_dfx.cost(dynamic_model, static_model)
            
        SHOW_BASE_TRANSFORM = False    
            
        print('RMSE is',RMSE)
        # PLOT trajectory
        if hasattr(forcing1D,'MLD'):
            fig, ax = plt.subplots(3,1,figsize = (10,10),constrained_layout=True,dpi=dpi)
        else:
            fig, ax = plt.subplots(2,1,figsize = (10,10),constrained_layout=True,dpi=dpi)
        if True:
            ax[0].plot((t0 + forcing1D.time)/oneday, U, c='k', lw=2, label='PAPA', alpha=0.3)
            ax[0].plot((t0 + forcing1D.time)/oneday, Ua, c='g', label='slab', alpha = 0.3)
            (Ut_nio,Vt_nio) = tools.my_fc_filter(mymodel.dt_forcing,U+1j*V, mymodel.fc)
            ax[0].plot((t0 + forcing1D.time)/oneday, Ut_nio, c='k', lw=2, label='PAPA at fc', alpha=1)
            (Unio,Vnio) = tools.my_fc_filter(mymodel.dt_forcing, Ua+1j*Va, mymodel.fc)
            ax[0].plot((t0 + forcing1D.time)/oneday, Unio, c='b', label='slab at fc')
        else:
            ax[0].plot((t0 + forcing1D.time)/oneday, U, c='k', lw=2, label='PAPA', alpha=1)
            ax[0].plot((t0 + forcing1D.time)/oneday, Ua, c='g', label='slab')
        ax[0].scatter((t0 + observations1D.time_obs)/oneday,Uo, c='r', label='obs', marker='x')
        ax[0].set_ylim([-0.6,0.6])
        #ax.set_xlim([15,25])
        #ax.set_title('RMSE='+str(np.round(RMSE,4))+' cost='+str(np.round(final_cost,4)))
        
        ax[0].set_ylabel('Ageo zonal current (m/s)')
        ax[0].legend(loc=1)
        # plot forcing
        ax[1].plot((t0 + forcing1D.time)/oneday, forcing1D.TAx, c='b', lw=2, label=r'$\tau_x$', alpha=1)
        ax[1].plot((t0 + forcing1D.time)/oneday, forcing1D.TAy, c='orange', lw=2, label=r'$\tau_y$', alpha=1)
        ax[1].set_ylabel('surface stress (N/m2)')
        ax[1].legend(loc=1)
        
        # plot MLD
        if hasattr(forcing1D,'MLD'):
            if hasattr(mymodel, 'dTK'):
                NdT = len(np.arange(mymodel.t0, mymodel.t1, mymodel.dTK))
                M = classic_slab.pkt2Kt_matrix(NdT, mymodel.dTK, mymodel.t0, mymodel.t1, mymodel.dt_forcing, base=mymodel.k_base)
                kt2D = classic_slab.kt_1D_to_2D(mymodel.pk, NdT, npk=len(mymodel.pk)//NdT*mymodel.nl)
                new_kt = np.dot(M,kt2D)
                myMLD = 1/np.exp(new_kt[:,0])/rho
                myR = np.transpose(np.exp(new_kt[:,1:])) * myMLD
                if SHOW_BASE_TRANSFORM and mymodel.k_base !='id':
                    M2 = classic_slab.pkt2Kt_matrix(NdT, mymodel.dTK, mymodel.t0, mymodel.t1, mymodel.dt_forcing, base='id')
                    new_kt2 = np.dot(M2,kt2D)
                    myMLD_cst = 1/np.exp(new_kt2[:,0])/rho
                    ax[2].plot((t0 + forcing1D.time)/oneday, myMLD_cst, label='no base transform')
                
                if SHOW_BASE_TRANSFORM:
                    fig2, ax2 = plt.subplots(1,1,figsize = (10,10),constrained_layout=True,dpi=dpi)
                    for k in range(M.shape[-1]):
                        ax2.plot((t0 + forcing1D.time)/oneday, M[:,k])
                        #ax2.plot(M2[:,k], ls='--')
            else:
                myMLD = 1/np.exp(mymodel.pk[0])/rho * np.ones(len(forcing1D.time))
                myR = np.stack( [np.exp(mymodel.pk[1:]) for _ in range(len(forcing1D.time))], axis=1) * myMLD
            ax[2].plot((t0 + forcing1D.time)/oneday, myMLD, label='estimated')
            ax[2].plot((t0 + forcing1D.time)/oneday, forcing1D.MLD, c='k', label='true')
            ax[2].set_ylabel('MLD (m)')
            ax[2].legend(loc=1)
        
            if False:
                ax2bis = ax[2].twinx()
                ax2bis.set_ylabel('r')
                lls = ['--','-.']
                for kr in range(len(myR)):
                    ax2bis.plot((t0 + forcing1D.time)/oneday, myR[kr],ls=lls[kr])
            ax[2].set_xlabel('Time (days)')
        fig.savefig(path_save_png+name_save+'_t'+str(int(mymodel.t0//oneday))+'_'+str(int(mymodel.t1//oneday))+'.png')
            
    plt.show()
            
            
            