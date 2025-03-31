"""
Here we want to find the optimal dTK for the vector pk.
"""


import numpy as np
import time as clock
import matplotlib.pyplot as plt
import sys
import os
import xarray as xr
sys.path.insert(0, '../src')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "True" # for jax
import jax.numpy as jnp

from models.classic_slab import *
import forcing
import observations
from constants import *
from functions_benchmark import benchmark_all
from my_var import ON_HPC

start = clock.time()



# ============================================================
# PARAMETERS
# ============================================================
# modify 'my_var' to switch from HPC to local      

# model parameters
Nl                  = 1         # number of layers for multilayer models
dTK                 = 10*oneday   # how much vectork K changes with time, basis change to exp
k_base              = 'gauss'   # base of K transform. 'gauss' or 'id'
AD_mode             = 'F'       # forward mode for AD 

# run parameters
t0                  = 0*oneday
dt_run              = 20*oneday
endt1               = 365*oneday
dt                  = 60.        # timestep of the model (s) 

L_lengths = [L for L in np.arange(200,360,dt_run)*oneday]


 #['jslab_kt_2D'] # 'all' #['jslab','jslab_kt','jslab_kt_2D','jslab_rxry','jslab_Ue_Unio','jslab_kt_Ue_Unio','jslab_kt','jslab_kt_2D','jslab_kt_Ue_Unio']
L_MODELS_TO_BENCHMARK = 'all'

L_all_models = ['jslab','jslab_kt','jslab_kt_2D','jslab_rxry','jslab_Ue_Unio','jslab_kt_Ue_Unio']
L_model_slab = ['jslab','jslab_kt','jslab_kt_2D','jslab_rxry']
L_model_slab_decoupled = ['jslab_Ue_Unio','jslab_kt_Ue_Unio']
L_model_kt = ['jslab_kt','jslab_kt_2D','jslab_kt_Ue_Unio']
L_model_2D = ['jslab_kt_2D']




# =================================
# Forcing, OSSE and observations
# =================================
# 1D
point_loc = [-50.,35.]
#point_loc = [-50.,46.] # should have more NIOs ?
point_loc = [-70., 35.]
# 2D
R = 1.0 # 20°x20° -> ~6.5Go of VRAM for grad
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

# Observations
period_obs          = oneday #86400      # s, how many second between observations  

# ============================================================
# END PARAMETERS
# ============================================================



if __name__ == "__main__": 
    
    file = []
    for ifile in range(len(name_regrid)):
        file.append(path_regrid+name_regrid[ifile])
        
    ### WARNINGS
    dsfull = xr.open_mfdataset(file)
    # warning about t1>length of forcing
    if endt1 > len(dsfull.time)*dt_forcing:
        print(f'You chose to run the model for t1={endt1//oneday} days but the forcing is available up to t={len(dsfull.time)*dt_forcing//oneday} days\n'
                        +f'I will use t1={len(dsfull.time)*dt_forcing//oneday} days')
        endt1 = len(dsfull.time)*dt_forcing
    ### END WARNINGS
        
   
    if L_MODELS_TO_BENCHMARK=='all':
        L_MODELS_TO_BENCHMARK = L_all_models
    
    
    
    
    
    for myt1 in L_lengths:
        t1 = float(myt1) # <- else t1 is treated as a control param by the JAX framework of the models
        print('')
        print('################')   
        print('# Benchmarking #')
        print(f'# period = {(t1-t0)/oneday} days')
        print('################') 

        print('-> getting forcing')
        forcing1D = forcing.Forcing1D(point_loc, t0, t1, dt_forcing, file)
        observations1D = observations.Observation1D(point_loc, period_obs, t0, t1, dt_OSSE, file)
        forcing2D = forcing.Forcing2D(dt_forcing, t0, t1, file, LON_bounds, LAT_bounds)
        observations2D = observations.Observation2D(period_obs, t0, t1, dt_OSSE, file, LON_bounds, LAT_bounds)
    
        call_args = t0, t1, dt
        
        
        L_models = []
        L_obs = []
        for nmodel in L_MODELS_TO_BENCHMARK:
            
            # control vector
            if nmodel in L_model_slab:
                pk = jnp.asarray([-11.31980127, -10.28525189])   
            elif nmodel in L_model_slab_decoupled:   
                pk = jnp.asarray([-11, -10, -10., -9.])  
                
            # extending K to Kt   
            if nmodel in L_model_kt:
                NdT = len(np.arange(t0, t1,dTK))
                mypk = kt_ini(pk, NdT)
            else:
                mypk = pk   
            
            # choosing between 1D and 2D     
            if nmodel in L_model_2D:
                frc, obs = forcing2D, observations2D
            else:
                frc, obs = forcing1D, observations1D
            
            # parameters
            TAx = jnp.asarray(frc.TAx)
            TAy = jnp.asarray(frc.TAy)
            fc = jnp.asarray(frc.fc)
            
            if nmodel=='jslab':
                mymodel= jslab(mypk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args)
            elif nmodel=='jslab_kt':
                mymodel = jslab_kt(mypk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args, k_base=k_base)
            elif nmodel=='jslab_kt_2D':
                mymodel = jslab_kt_2D(mypk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args,use_difx=False, k_base=k_base)
            elif nmodel=='jslab_rxry':
                mymodel = jslab_rxry(mypk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args)
            elif nmodel=='jslab_Ue_Unio':
                mymodel = jslab_Ue_Unio(mypk, TAx, TAy, fc, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args)
            elif nmodel=='jslab_kt_Ue_Unio':
                mymodel = jslab_kt_Ue_Unio(mypk, TAx, TAy, fc, dTK, dt_forcing, nl=1, AD_mode=AD_mode, call_args=call_args, use_difx=False)
            else:
                raise Exception(f'the model {nmodel} is not recognized, aborting ...')
                        
            L_models.append(mymodel)
            L_obs.append(obs)

        benchmark_all(L_models, L_obs, Nexec=10)
        
    