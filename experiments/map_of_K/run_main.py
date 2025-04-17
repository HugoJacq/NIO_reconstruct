"""
This script is meant to plot maps of the parameters from models jslab_kt_2D and junsteak_kt_2D

Is there any correlation between map of K and maps of physical feature (such as gradient of geostrophic)
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
import jax.numpy as jnp
import equinox as eqx

# THIS NEEDS TO BE MERGED INTO A CLEAN SOLUTION
from Listes_models import *

import forcing
import observations
import inv
import tools
from basis import kt_ini
from constants import *


from functions import iter_bounds_mapper
from map_of_pk import compute_and_save_pk



# ===========================================================================
# PARAMETERS
# ===========================================================================

# model parameters    
t0                  = 60*oneday    # start day 
t1                  = 100*oneday    # end day
dt                  = 60.           # timestep of the model (s) 
call_args = t0, t1, dt
args_model = {'dTK' : 10*oneday,    # how much vectork K changes with time, basis change to 'k_base'      
              'Nl'  : 2             # number of layers for multilayer models
              }
extra_args = {'AD_mode':'F',        # forward mode for AD (for diffrax' diffeqsolve)
            'use_difx':False,       # use diffrax solver to time integrate
            'k_base':'gauss'}       # base of K transform. 'gauss' or 'id'

# What to test
mini_args = {'maxiter':2,           # max number of iteration
             }


ON_PAPA             = False      # use PAPA station data, only for 1D models
FILTER_AT_FC        = False      # minimize filtered ageo current with obs if model has this option
IDEALIZED_RUN       = False       # try the model on a step wind stress

# Switches
L_model_to_test             = ['junsteak_kt_2D_adv'] # L_all

# PLOT
dpi=200
path_save_png = './png_maps/'

# =================================
# Forcing, OSSE and observations
# =================================
# # 1D
# point_loc = [-47.4,34.6] # march 8th 0300, t0=60 t1=100
# # 2D
R = 2.5 # 5.0 
# LON_bounds = [point_loc[0]-R,point_loc[0]+R]
# LAT_bounds = [point_loc[1]-R,point_loc[1]+R]
# Forcing
dt_forcing          = onehour      # forcing timestep
# Croco data
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


# Observations
period_obs          = oneday #86400      # s, how many second between observations  

# ===========================================================================
# END PARAMETERS
# ===========================================================================

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
             
    print('Running: tests_models.py')   
    print('**************')
    print('full Croco domain')
    print(f"if multi layer, nl={args_model['Nl']}")
    print('**************\n')
    
    
    model_name = 'jslab_kt_2D'
    
    if model_name in L_unsteaks:
        module_name = 'unsteak'
    elif model_name in L_slabs:
        module_name = 'classic_slab'
    
    k = 0
    
    print(f'LON min {np.round(np.min(dsfull.lon.values),2)}, max {np.round(np.max(dsfull.lon.values),2)}')
    print(f'LAT min {np.round(np.min(dsfull.lat.values),2)}, max {np.round(np.max(dsfull.lat.values),2)}')
    
    
    fig, ax = plt.subplots(1,1,figsize = (10,6),constrained_layout=True,dpi=dpi)
    
    ax.scatter(np.round(np.min(dsfull.lon.values),2),np.round(np.min(dsfull.lat.values),2))
    ax.add_patch(matplotlib.patches.Rectangle(
                                    (np.round(np.min(dsfull.lon.values),2),np.round(np.min(dsfull.lat.values),2)), 
                                    np.round(np.max(dsfull.lon.values),2) - np.round(np.min(dsfull.lon.values),2), 
                                    np.round(np.max(dsfull.lat.values),2) - np.round(np.min(dsfull.lat.values),2),
                                    fill=False,
                                    edgecolor='k'
                                    ),
                 )
    ax.set_title(f'tile map for side={2*R}°')    
    ax.set_aspect(1)
    for (point_loc, LON_bounds, LAT_bounds) in iter_bounds_mapper(R, dx=0.1, lon=dsfull.lon.values, lat=dsfull.lat.values):
        """
        """
        
        # print some infos       
        # print(f'{k} center of tile is {np.round(point_loc[0],2)}°E,{np.round(point_loc[1],2)}°N')
        # print(f'     LON_bounds {np.round(LON_bounds[0],2)} -> {np.round(LON_bounds[1],2)}')
        # print(f'     LAT_bounds {np.round(LAT_bounds[0],2)} -> {np.round(LAT_bounds[1],2)}')
        k=k+1
        ax.add_patch(matplotlib.patches.Rectangle(
                                    (LON_bounds[0],LAT_bounds[0]), 
                                    LON_bounds[1] - LON_bounds[0], 
                                    LAT_bounds[1] - LAT_bounds[0],
                                    fill=False,
                                    edgecolor='r'
                                    ))
        #ax.scatter(point_loc[0],point_loc[1],c='r',marker='x')
        ax.annotate(str(k), (point_loc[0],point_loc[1]), c='r')
        # # forcing
        # if model_name in L_1D_models:
        #     myforcing = forcing.Forcing1D(point_loc, t0, t1, dt_forcing, file)
        #     myobservation = observations.Observation1D(point_loc, period_obs, t0, t1, dt_forcing, file)
        # elif model_name in L_2D_models:
        #     myforcing = forcing.Forcing2D(dt_forcing, t0, t1, file, LON_bounds, LAT_bounds)
        #     myobservation = observations.Observation2D(period_obs, t0, t1, dt_forcing, file, LON_bounds, LAT_bounds)
            
        # def model_instanciation(model_name, forcing, args_model, call_args, extra_args):
        #     """
        #     Wrapper of every model instanciation.
            
            
        #     TBD: doc of this, with typing
        #     """
        #     t0, t1, dt = call_args
        #     dTK = args_model['dTK']
        #     Nl = args_model['Nl']
            
        #     # MODELS FROM MODULE classic_slab
        #     if model_name=='jslab':
        #         pk = jnp.asarray([-11., -10.]) 
        #         model = classic_slab.jslab(pk, forcing, call_args, extra_args)
                
        #     elif model_name=='jslab_fft':
        #         pk = jnp.asarray([-11., -10.]) 
        #         model = classic_slab.jslab_fft(pk, forcing, call_args, extra_args)
                
        #     elif model_name=='jslab_kt':
        #         pk = jnp.asarray([-11., -10.])   
        #         NdT = len(np.arange(t0, t1,dTK))
        #         pk = kt_ini(pk, NdT)
        #         model = classic_slab.jslab_kt(pk, dTK, forcing, call_args, extra_args)
                
        #     elif model_name=='jslab_kt_2D':
        #         pk = jnp.asarray([-11., -10.])   
        #         NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        #         pk = kt_ini(pk, NdT)
        #         model = classic_slab.jslab_kt_2D(pk, dTK, forcing, call_args, extra_args)
                
        #     elif model_name=='jslab_rxry':
        #         pk = jnp.asarray([-11., -10., -9.]) 
        #         model = classic_slab.jslab_rxry(pk, forcing, call_args, extra_args)
                
        #     elif model_name=='jslab_Ue_Unio':
        #         TA = forcing.TAx + 1j*forcing.TAy
        #         TAx_f,TAy_f = tools.my_fc_filter( forcing.dt_forcing, TA, forcing.fc)
        #         forcing.TAx = TAx_f
        #         forcing.TAy = TAy_f
        #         pk = jnp.asarray([-11., -10.]) 
        #         model = classic_slab.jslab_Ue_Unio(pk, forcing, call_args, extra_args)
                
        #     elif model_name=='jslab_kt_Ue_Unio':
        #         TA = forcing.TAx + 1j*forcing.TAy
        #         TAx_f,TAy_f = tools.my_fc_filter( forcing.dt_forcing, TA, forcing.fc)
        #         forcing.TAx = TAx_f
        #         forcing.TAy = TAy_f
        #         pk = jnp.asarray([-11., -10.])   
        #         NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        #         pk = kt_ini(pk, NdT)
        #         model = classic_slab.jslab_kt_Ue_Unio(pk, dTK, forcing, call_args, extra_args)
                
        #     elif model_name=='jslab_kt_2D_adv':
        #         pk = jnp.asarray([-11., -10.])   
        #         NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        #         pk = kt_ini(pk, NdT)
        #         model = classic_slab.jslab_kt_2D_adv(pk, dTK, forcing, call_args, extra_args)
                
        #     elif model_name=='jslab_kt_2D_adv_Ut':
        #         pk = jnp.asarray([-11., -10.])   
        #         NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        #         pk = kt_ini(pk, NdT)
        #         model = classic_slab.jslab_kt_2D_adv_Ut(pk, dTK, forcing, call_args, extra_args)
                
            
            
        #     # # MODELS FROM MODULE unsteak
        #     elif model_name=='junsteak':
        #         if Nl==1:
        #             pk = jnp.asarray([-11.31980127, -10.28525189])    
        #         elif Nl==2:
        #             pk = jnp.asarray([-10.,-10., -9., -9.])   
        #         model = unsteak.junsteak(pk, forcing, Nl, call_args, extra_args)   
                
        #     elif model_name=='junsteak_kt':
        #         if Nl==1:
        #             pk = jnp.asarray([-11.31980127, -10.28525189])    
        #         elif Nl==2:
        #             pk = jnp.asarray([-10.,-10., -9., -9.]) 
        #         NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        #         pk = kt_ini(pk, NdT)
        #         model = unsteak.junsteak_kt(pk, dTK, forcing, Nl, call_args, extra_args)   
                
        #     elif model_name=='junsteak_kt_2D':
        #         if Nl==1:
        #             pk = jnp.asarray([-11.31980127, -10.28525189])    
        #         elif Nl==2:
        #             pk = jnp.asarray([-10.,-10., -9., -9.]) 
        #         NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        #         pk = kt_ini(pk, NdT)
        #         model = unsteak.junsteak_kt_2D(pk, dTK, forcing, Nl, call_args, extra_args)   
            
        #     elif model_name=='junsteak_kt_2D_adv':
        #         if Nl==1:
        #             pk = jnp.asarray([-11.31980127, -10.28525189])    
        #         elif Nl==2:
        #             pk = jnp.asarray([-10.,-10., -9., -9.]) 
        #         NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        #         pk = kt_ini(pk, NdT)
        #         model = unsteak.junsteak_kt_2D_adv(pk, dTK, forcing, Nl, call_args, extra_args)   
                
        #     else:
        #         raise Exception(f'You want to test the mode {model_name} but it is not recognized')

            
        #     return model


        # # model initialization
        # mymodel = model_instanciation(model_name, myforcing, args_model, call_args, extra_args)
        # var = inv.Variational(mymodel, myobservation, filter_at_fc=False)
        
        
        # model_name = type(mymodel).__name__
        # save_name = ''
        
        # compute_and_save_pk(mymodel, var, mini_args, save_name)
    
    fig.savefig(path_save_png+f'tile_map_side{2*R}.png')
    
    
    
    # plots
    
    
    plt.show()
        
        
        
        
    """
    I need to make a map of LAT_bounds and LON_bounds, then apply the minimization on each to find the vector K.
    1 month can be targeted, and we can show the (time) mean of a component of K.
    
    dTK = 10days -> 3 values for 1 month
    
    jslab_kt_2D -> 2 components
    junsteak_kt_2D -> 4 components
    
    SAVE THE K VALUES, as the minimize step of all tiles will be quite long
    
    To compare with: 
    maps at the same resolution (2*R) of physical quantities:
    - dUdx
    - MLD
    - Tau
    
    """
    
    # if model_name in L_1D_models:
    #     myforcing = forcing.Forcing1D(point_loc, t0, t1, dt_forcing, file)
    #     myobservation = observations.Observation1D(point_loc, period_obs, t0, t1, dt_forcing, file)
    # elif model_name in L_2D_models:
    #     myforcing = forcing.Forcing2D(dt_forcing, t0, t1, file, LON_bounds, LAT_bounds)
    #     myobservation = observations.Observation2D(period_obs, t0, t1, dt_forcing, file, LON_bounds, LAT_bounds)
    
    
    
    