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

from map_of_K.functions import compute_and_save_pk
from tests_models.tests_functions import model_instanciation

import forcing
import observations
import inv
from constants import oneday, onehour
from tools import rotary_spectra, rotary_spectra_2D
from Listes_models import L_2D_models, L_nlayers_models
# ===========================================================================
# PARAMETERS
# ===========================================================================

NORM_FREQ = False

# model parameters    
t0                  = 60*oneday     # start day 
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
mini_args = {'maxiter':100,           # max number of iteration
             }

# Switches
L_model_to_test             = ['jslab_kt_2D','junsteak_kt_2D','junsteak_kt_2D_adv'] # jslab_kt_2D junsteak_kt_2D junsteak_kt_2D_adv
SAVE_PKs                    = False
SHOW_INFO                   = False                 # if SAVE_PKs
path_save_pk = './pk_save/'

# location
point_loc = [-47.4,34.6]    # event of march 8th 0300, t0=60 t1=100
R = 5                       # degrees
LON_bounds = [point_loc[0]-R,point_loc[0]+R]
LAT_bounds = [point_loc[1]-R,point_loc[1]+R]

# figures
path_save_png = './png_scores/'

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
    ### END WARNINGS
    
    myforcing = forcing.Forcing2D(dt_forcing, t0, t1, file, LON_bounds, LAT_bounds)
    myobservation = observations.Observation2D(period_obs, t0, t1, dt_forcing, file, LON_bounds, LAT_bounds)
    
    
    ### SAVING PKs
    if SAVE_PKs:
        print(f'        center of tile is {np.round(point_loc[0],2)}°E,{np.round(point_loc[1],2)}°N')
        print(f'        LON_bounds {np.round(LON_bounds[0],2)} -> {np.round(LON_bounds[1],2)}')
        print(f'        LAT_bounds {np.round(LAT_bounds[0],2)} -> {np.round(LAT_bounds[1],2)}\n')
        # forcing
        
        tile_infos = 0, point_loc, LON_bounds, LAT_bounds
        print('* Searching for the pk ...')
        for model_name in L_model_to_test:
            print('     '+model_name)
            # model initialization
            mymodel = model_instanciation(model_name, myforcing, args_model, call_args, extra_args)
            var = inv.Variational(mymodel, myobservation, filter_at_fc=False)
                            
            compute_and_save_pk(mymodel, var, mini_args, tile_infos, path_save_pk)
    
    ### PLOTS
    
    # run a trajectory on the domain.
    # apply rotatry spectra on each cell, then average all of them.
    # plot spectra and score, Rmse
    
    for model_name in L_model_to_test:  
        
        mymodel = model_instanciation(model_name, myforcing, args_model, call_args, extra_args)
        # replacing control parameters
        pk = np.load(path_save_pk + model_name + '/0.npy') 
        mymodel = eqx.tree_at(lambda t:t.pk, mymodel, pk.flatten())
        
        sol = mymodel()
        if model_name in L_nlayers_models:
            sol = sol[0][:,0,:,:], sol[1][:,0,:,:]
        
        truth = myobservation.U, myobservation.V
        nt, ny, nx = sol[0].shape        
        
        RMSE = np.mean( np.sqrt( (sol[0]-truth[0])**2 + (sol[1]-truth[1])**2))
        
        ff, CWr, ACWr, CWe, ACWe = rotary_spectra_2D(1., sol[0], sol[1], truth[0], truth[1])
        if NORM_FREQ:
            mean_fc = 2*2*np.pi/86164*np.sin(point_loc[1]*np.pi/180)*onehour/(2*np.pi)
            xtxt = r'f/$f_c$'
        else:
            mean_fc = 1
            xtxt = 'h-1'
        fig, axs = plt.subplots(2,1,figsize=(7,6), gridspec_kw={'height_ratios': [4, 1.5]})
        axs[0].loglog(ff/mean_fc,CWr, c='k', label='reference')
        axs[0].loglog(ff/mean_fc,CWe, c='b', label='error (model - truth)')
        #axs[0].axis([2e-3,2e-1, 1e-4,2e0])
        axs[0].grid('on', which='both')
        axs[1].set_xlabel(xtxt)
        axs[0].legend()
        axs[0].set_ylabel('Clockwise PSD (m2/s2)')
        axs[0].title.set_text(model_name +f', RMSE={np.round(RMSE,5)}')

        axs[1].semilogx(ff/mean_fc,(1-CWe/CWr)*100, c='b', label='Reconstruction Score')
        #axs[1].axis([2e-3,2e-1,0,100])
        axs[1].set_ylim([0,100])
        axs[1].grid('on', which='both')
        axs[1].set_ylabel('Scores (%)')
        fig.savefig(path_save_png + model_name + '_diag.png')
        
        xtime = np.arange(t0,t1,dt_forcing)/oneday
        fig, ax = plt.subplots(1,1 , figsize=(10,6), constrained_layout=True)
        ax.plot(xtime, sol[0][:,ny//2,nx//2], label=model_name)
        ax.plot(xtime, truth[0][:,ny//2,nx//2], label='truth')
        ax.legend()
        ax.set_xlabel('time (days)')
        ax.set_ylabel('Ageo current m/s')
    
    print(f'Total execution time : {clock.time()-time0}')
    plt.show()