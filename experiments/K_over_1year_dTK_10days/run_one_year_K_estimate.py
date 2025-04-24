import os
import sys
import time as clock


sys.path.insert(0, '../../src')
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import equinox as eqx
from pprint import pprint

from map_of_K.functions import compute_and_save_pk
from tests_models.tests_functions import model_instanciation

import forcing
import observations
import inv
from constants import oneday, onehour
# from Listes_models import L_2D_models, L_nlayers_models

"""
This script is looking at how the control parameters evolve along the year, for several models.
As the forcing for one year cannot fit in GPU memory, we split the minimisation in several batches of time,
    with length 'twindow'. They overlap with 'toverlap'.
    
Control parameters (the PKs) are saved to allow for easy plotting.
"""



# ===========================================================================
# PARAMETERS
# ===========================================================================

# model parameters    
t0                  = 0*oneday     # start day 
t1                  = 130*oneday    # end day
twindow             = 60*oneday    # max of time length that fit in memory
toverlap            = 20*oneday     # number of seconds for the overlap
dt                  = 60.           # timestep of the model (s) 

args_model = {'dTK' : 10*oneday,    # how much vectork K changes with time, basis change to 'k_base'      
              'Nl'  : 2             # number of layers for multilayer models
              }
extra_args = {'AD_mode':'F',        # forward mode for AD (for diffrax' diffeqsolve)
            'use_difx':False,       # use diffrax solver to time integrate
            'k_base':'gauss'}       # base of K transform. 'gauss' or 'id'

# What to test
mini_args = {'maxiter':100,           # max number of iteration
             }

# Switches
L_model_to_test             = ['jslab_kt_2D'] # jslab_kt_2D junsteak_kt_2D junsteak_kt_2D_adv
SAVE_PKs                    = False
SHOW_INFO                   = False                 # if SAVE_PKs
path_save_pk = './pk_save/'

# location
point_loc = [-47.4,34.6]    # event of march 8th 0300, t0=60 t1=100
R = 5                       # degrees
LON_bounds = [point_loc[0]-R,point_loc[0]+R]
LAT_bounds = [point_loc[1]-R,point_loc[1]+R]

# figures
path_save_png = './png_K_with_time/'

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
    time0 = clock.time()
    file = []
    for ifile in range(len(name_regrid)):
        file.append(path_regrid+name_regrid[ifile])
    
    dsfull = xr.open_mfdataset(file)
    lon = dsfull.lon.values
    lat = dsfull.lat.values

    ### WARNINGS
    # warning about t1>length of forcing
    if t1 > len(dsfull.time)*dt_forcing:
        print(f'You chose to run the model for t1={t1//oneday} days but the forcing is available up to t={len(dsfull.time)*dt_forcing//oneday} days\n'
                        +f'I will use t1={len(dsfull.time)*dt_forcing//oneday} days')
        t1 = len(dsfull.time)*dt_forcing
    
    if (t1-t0)%args_model['dTK'] !=0:
        raise Exception(f'Your choice of dTK ({args_model["dTK"]/oneday} days) is not a divider of t1-t0 ({(t1-t0)/oneday} days)\nPlease  retry')
    ### END WARNINGS
    
    
    
    ### SAVING PKs
    if SAVE_PKs:
        print('* Estimate of pk ...')
        print(f' center of tile is {np.round(point_loc[0],2)}°E,{np.round(point_loc[1],2)}°N')
        print(f' LON_bounds {np.round(LON_bounds[0],2)} -> {np.round(LON_bounds[1],2)}')
        print(f' LAT_bounds {np.round(LAT_bounds[0],2)} -> {np.round(LAT_bounds[1],2)}\n')
        
        
        tstart = 0.
        tend = twindow
        k = 0
        while tend<= t1 and (tend-tstart>toverlap):
            print(f'Working on window n°{k} from {tstart/oneday} days to {tend/oneday}')                      
            # forcing
            myforcing = forcing.Forcing2D(dt_forcing, tstart, tend, file, LON_bounds, LAT_bounds)
            myobservation = observations.Observation2D(period_obs, tstart, tend, dt_forcing, file, LON_bounds, LAT_bounds)
            call_args = tstart, tend, dt
            
            tile_infos = k, point_loc, LON_bounds, LAT_bounds
            for model_name in L_model_to_test:
                print('     '+model_name)
                # model initialization
                mymodel = model_instanciation(model_name, myforcing, args_model, call_args, extra_args)
                var = inv.Variational(mymodel, myobservation, filter_at_fc=False)
                                
                compute_and_save_pk(mymodel, var, mini_args, tile_infos, path_save_pk, save_link_file=False)

                      
            k=k+1
            # finding next time window, according to overlap
            tstart = tend - toverlap
            tend = tstart + twindow
            
            # special case at t1
            if (tend>t1):
                tend = t1
            
            
    ### PLOTS
    print('* Plotting')
    for model_name in L_model_to_test:  
        # replacing control parameters
        dTK = args_model['dTK']
        
        L_pk = []
        npy_path = path_save_pk + model_name        
        number_of_files = len(os.listdir(npy_path+'/'))
        for k in range(number_of_files):
            L_pk.append(np.load(npy_path + f'/{k}.npy'))
        
        
        pprint([L_pk[k].shape for k in range(number_of_files)])
        ipk = 0
        k=0
        tstart = t0
        tend = twindow
        
        fig,ax = plt.figure(figsize=(10, 5),constrained_layout=True,dpi=200)
        while k<len(L_pk):
            pk = L_pk[k]
            
            x_scatter = np.linspace(tstart + dTK/2, tend - dTK/2, pk.shape[0] )/oneday
            ax.scatter()
            
            # here use pkt2M function ! to transform to the basis.
            # either gauss or id
                
            
            
        