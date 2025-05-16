"""

Comparing models with and without advection term

models to be compared : slab_kt_2D vs jslab_kt_2D_adv
                    unsteak_kt_2D vs unsteak_kt_2D_adv

A figure with RMSE(t), Rotary spectra(t), PSD score(t)
   - slab model with -rU
   - unsteak 2 couches
   - slab avec -rU avec adv
   - unsteak 2 couches avec adv

Influence of the advection term on the reconstructed trajectory near an eddy:
we expect to see improvment on the phase of the NIOs.

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
import jax
import jax.numpy as jnp
import equinox as eqx

from models import classic_slab, unsteak
from basis import kt_ini
import forcing
import inv
import observations
import tools
from constants import *
from Listes_models import *

from functions import save_output_as_nc
# ============================================================
# PARAMETERS
# ============================================================

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
MINIMIZE            = True      # Does the model converges to a solution ?
maxiter             = 100        # if MINIMIZE: max number of iteration
SAVE_AS_NC          = True      # for 2D models

# PLOT
dpi=200
path_save_png = './pngs/'
path_save_models = './saved_models/'
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
os.system('mkdir -p '+path_save_models)
os.system('mkdir -p '+path_save_output)

if MINIMIZE: SAVE_AS_NC=True # if we find new parameters, rerun the reconstruction

if __name__ == "__main__": 

    file = []
    for ifile in range(len(name_regrid)):
        file.append(path_regrid+name_regrid[ifile])
    dsfull = xr.open_mfdataset(file)
    indx = tools.nearest(dsfull.lon.values,point_loc[0])
    indy = tools.nearest(dsfull.lat.values,point_loc[1])
    txt_location = f'{dsfull.lon.values[indx]}°E, {dsfull.lat.values[indy]}°N'

    print('Running: run_effect_advection.py')   
    print('**************')
    print('Location is '+txt_location)
    print(f'2D slice is {LON_bounds}°E {LAT_bounds}°N')
    print(f'if multi layer, nl={Nl}')
    print('**************\n')
    
    
    # forcing and obs initialization
    myforcing = forcing.Forcing2D(dt_forcing, t0, t1, file, LON_bounds, LAT_bounds)
    myobservation = observations.Observation2D(period_obs, t0, t1, dt_OSSE, file, LON_bounds, LAT_bounds)
    call_args = t0, t1, dt
    args_model = {'dTK':dTK, 'Nl':Nl}
    
    dict_models = {}
    
    # initialize slab models
    pk = jnp.asarray([-11., -10.])   
    NdT = len(np.arange(t0, t1, dTK)) # int((t1-t0)//dTK) 
    pk = kt_ini(pk, NdT)
    # dict_models['myslab_kt_2D'] = classic_slab.jslab_kt_2D(pk, dTK, myforcing, call_args, extra_args)
    # dict_models['myslab_kt_2D_adv'] = classic_slab.jslab_kt_2D_adv(pk, dTK, myforcing, call_args, extra_args)
    
    # initialize unsteak models
    if Nl==1:
        pk = jnp.asarray([-11.31980127, -10.28525189])    
    elif Nl==2:
        pk = jnp.asarray([-15.,-10., -10., -10.]) 
    NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
    pk = kt_ini(pk, NdT)
    # dict_models['myunsteak_kt_2D'] = unsteak.junsteak_kt_2D(pk, dTK, myforcing, Nl, call_args, extra_args)   
    dict_models['myunsteak_kt_2D_adv2l'] = unsteak.junsteak_kt_2D_adv(pk, dTK, myforcing, Nl, call_args, extra_args, alpha=[1.,1.]) 
    dict_models['myunsteak_kt_2D_adv1l'] = unsteak.junsteak_kt_2D_adv(pk, dTK, myforcing, Nl, call_args, extra_args, alpha=[1.,0.])    
    
    # list_models = [myslab_kt_2D, myslab_kt_2D_adv, myunsteak_kt_2D, myunsteak_kt_2D_adv2l, myunsteak_kt_2D_adv1l]
    ################################################
    # STEP 1:
    #
    # Find the best parameters for each models
    ################################################
    if MINIMIZE:
        print('* starting to minimize')
        tstart = clock.time()
        for model_name, mymodel in dict_models.items():
            # model_name = type(mymodel).__name__
            print(f'\n     working on {model_name}..')
            t2 = clock.time()
            var = inv.Variational(mymodel, myobservation)
            mymodel, _ = var.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            eqx.tree_serialise_leaves(path_save_models+f'best_{model_name}.pt', mymodel)
            print(f'        time, minimize {model_name} = {clock.time()-t2}')
        print(f'time, minimize all = {clock.time()-tstart}')

    ################################################
    # STEP 2:
    #
    # Load back the models with their parameters
    #   and save the reconstructed currents as .nc
    ################################################  
    for model_name, mymodel in dict_models.items():
        model_name = type(mymodel).__name__
        dict_models[model_name] = eqx.tree_deserialise_leaves(path_save_models+f'best_{model_name}.pt', mymodel)
    
    if SAVE_AS_NC:
        print('* saving each model outputs')
        for model_name, mymodel in dict_models.items():
            # model_name = type(model).__name__
            name_save = model_name+'_'+namesave_loc
            save_output_as_nc(mymodel, myforcing, LON_bounds, LAT_bounds, name_save, path_save_output)

    ################################################
    # STEP 3:
    #
    # Load .nc datasets for analysis, then plots
    ################################################
    
    
    
    datas = {"slab_kt_2D":xr.open_mfdataset(path_save_output+"jslab_kt_2D_-49.0E_39.0N.nc"),
             "slab_kt_2D_adv":xr.open_mfdataset(path_save_output+"jslab_kt_2D_adv_-49.0E_39.0N.nc"),
             "junsteak_kt_2D":xr.open_mfdataset(path_save_output+"junsteak_kt_2D_-49.0E_39.0N.nc"),
             "slab_kt_2D_adv":xr.open_mfdataset(path_save_output+"jslab_kt_2D_adv_-49.0E_39.0N.nc"),
             "junsteak_kt_2D_adv":xr.open_mfdataset(path_save_output+"junsteak_kt_2D_adv_-49.0E_39.0N.nc")}
    
    
    
    
    
    
    
"""Plot: 

- when plotting a trajectory, plot the stress also
- plot of Ug on the area, shouldnt change too much on 10 days! -> help locate the eddy of interest.

- plot trajectories of reconstructed current for each model at the center of the eddy (or on the side ?)
- plot RMSE of this trajectory, plot mean RMSE over the XY patch
- plot spectra and score over the XY patch


"""