"""
Here we want to find the optimal dTK for the vector pk.
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

from models.classic_slab import jslab, jslab_kt, jslab_kt_2D, kt_ini, kt_1D_to_2D, pkt2Kt_matrix
import forcing
import inv
import observations
#from tests_functions import run_forward_cost_grad, plot_traj_1D, plot_traj_2D
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
maxiter             = 100         # max number of iteration
PLOT_TRAJ           = True



TEST_SLAB_KT                = True

# PLOT
dpi=200
path_save_png = './png_opti_dTK/'

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
        
    
    ### WARNINGS
    dsfull = xr.open_mfdataset(file)
    # warning about t1>length of forcing
    if t1 > len(dsfull.time)*dt_forcing:
        print(f'You chose to run the model for t1={t1//oneday} days but the forcing is available up to t={len(dsfull.time)*dt_forcing//oneday} days\n'
                        +f'I will use t1={len(dsfull.time)*dt_forcing//oneday} days')
        t1 = len(dsfull.time)*dt_forcing
    ### END WARNINGS
    
    
    
    if TEST_SLAB_KT:
        # initial control vector
        pkini = jnp.asarray([-11.31980127, -10.28525189])          
        call_args = t0, t1, dt
        # list of dTK to test
        list_dTK = [int(dTK) for dTK in np.arange(10,0,-1)*oneday] # 60
        
        # for 1 batch of 6 months, minimize the vector_k for each dTK from list_dTK, plot the cost function at the end of each minimisation
        # for 1 batch of the other 6 months, plot the cost function with the vector_k associated with each dTK
        
        # plot 1 should be always decreasing
        # plot 2 will decrease then increase, optimal is at the change of slope !
                
        L_cost1 = []
        L_cost2 = []
        converged = []
        a_lot_of_it = []
        
        for dTK in list_dTK:
            
            NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
            pk = kt_ini(pkini, NdT)
            print('')
            print('*** dTK (days) =',dTK/oneday)
            # minimisation
            forcing1D = forcing.Forcing1D(point_loc, t0, t1, dt_forcing, file)
            observations1D = observations.Observation1D(point_loc, period_obs, t0, t1, dt_OSSE, file)
            TAx = np.asarray(forcing1D.TAx)
            TAy = np.asarray(forcing1D.TAy)
            fc = np.asarray(forcing1D.fc)
            mymodel = jslab_kt(pk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args, k_base=k_base)
            var_dfx = inv.Variational(mymodel,observations1D)
            
            mymodel, res = var_dfx.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True) 
            
            if res.nit<maxiter//2:
                converged.append(1)
            elif (res.nit>maxiter//2) and (res.nit<maxiter):
                converged.append(2)
            else:
                converged.append(0)
            
            dynamic_model, static_model = inv.my_partition(mymodel)
            final_cost = var_dfx.cost(dynamic_model, static_model)
            
            
            pk = mymodel.pk
            L_cost1.append(final_cost)
            
            # no minimisation, reuse the vector_k
            forcing1D = forcing.Forcing1D(point_loc, t0, t1, dt_forcing, file)
            observations1D = observations.Observation1D(point_loc, dt_forcing, t0, t1, dt_OSSE, file) # <- here I use period_obs=dt_forcing
            TAx = jnp.asarray(forcing1D.TAx)
            TAy = jnp.asarray(forcing1D.TAy)
            fc = jnp.asarray(forcing1D.fc)
            mymodel = jslab_kt(pk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args, k_base=k_base)
            var_dfx = inv.Variational(mymodel,observations1D)
            
            dynamic_model, static_model = inv.my_partition(mymodel)
            final_cost = var_dfx.cost(dynamic_model, static_model)
            L_cost2.append(final_cost)

        
        L_cost1 = np.asarray(L_cost1)
        L_cost2 = np.asarray(L_cost2)
        list_dTK = np.asarray(list_dTK)
        
        cross_converged = np.where(np.array(converged)==1, L_cost1, np.nan)
        cross_a_lot_of_it = np.where(np.array(converged)==2, L_cost1, np.nan)
        cross_not_converged = np.where(np.isnan(cross_converged+cross_a_lot_of_it), np.nan, L_cost1)

        fig, ax = plt.subplots(1,1,figsize = (5,5),constrained_layout=True,dpi=dpi)
        ax.plot(list_dTK/oneday, L_cost1, c='k', label='cost_batch1')
        ax.plot(list_dTK/oneday, L_cost2, c='b', label='cost_batch2')
        ax.scatter(list_dTK/oneday, cross_a_lot_of_it, c='orange', marker='x')
        ax.scatter(list_dTK/oneday, cross_not_converged, c='r', marker='x')
        ax.scatter(list_dTK/oneday, cross_converged, c='g', marker='x')
        ax.set_xlim([list_dTK[0]/oneday,list_dTK[-1]/oneday])
        ax.set_xlabel('dTK (days)')
        ax.set_ylabel('cost')
        ax.legend()
        fig.savefig(path_save_png+'cost_dTK_'+str(int(list_dTK[0]/oneday))+'_to_'+str(int(list_dTK[-1]/oneday))+'.png')
        
    plt.show()