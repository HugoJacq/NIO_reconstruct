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
# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
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
from constants import distance_1deg_equator
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
dt                  = 60.           # timestep of the model (s) 

# What to test
MINIMIZE            = False     # Does the model converges to a solution ?
maxiter             = 100       # if MINIMIZE: max number of iteration
SAVE_AS_NC          = False      

# PLOT
dpi=200
path_save_png = './pngs/'
path_save_models = './saved_models/'
path_save_output = './saved_outputs/'

PLOT_TRAJ = False
PLOT_RMSE = False
PLOT_DIAGFREQ = True
PLOT_SNAPSHOT = False

# =================================
# Forcing, OSSE and observations
# =================================
point_loc = [-49.,39.]  # march 8th 0300, t0=60,65 t1=100,75, centered on an eddy
t0, t0_plot = 60*oneday, 65*oneday   # start day 
t1, t1_plot = 100*oneday, 75*oneday  # end day
location_to_plot = [-49.5,39.5]

# point_loc = [-60.,37.]  # february 26th t0=50 ,t1=30
# t0, t0_plot = 50*oneday, 50*oneday   # start day 
# t1, t1_plot = 80*oneday, 80*oneday  # end day
# location_to_plot = [-67.,38.]

# point_loc = [-46.5,40.]  # feb 1rst, t0=32 t1=72, centered on a cyclonic eddy
# t0, t0_plot = 60*oneday, 65*oneday   # start day 
# t1, t1_plot = 100*oneday, 75*oneday  # end day
# location_to_plot = [-49.,39.]

"""
cyclone:
    point_loc = [-49.,39.]  # march 8th 0300, t0=60,65 t1=100,75, centered on an eddy
    t0, t0_plot = 60*oneday, 65*oneday   # start day 
    t1, t1_plot = 100*oneday, 75*oneday  # end day
    indt = 239
    location = [-49.5,39.5]
    
    point_loc = [-46.5,40.]  # feb 23rd, t0=32 t1=72, centered on a cyclonic eddy
    t0, t0_plot = 32*oneday, 32*oneday   # start day 
    t1, t1_plot = 72*oneday, 72*oneday  # end day
    indt = 523
    location = [-49.,39.]
    
    
anticyclone: ?

    
"""


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

# Observations
period_obs          = oneday #86400      # s, how many second between observations  

# ============================================================
# END PARAMETERS
# ============================================================
namesave_loc = f'ts{int(t0/oneday)}_te{int(t1/oneday)}_{point_loc[0]}E_{point_loc[1]}N'
namesave_loc_area = f'ts{int(t0/oneday)}_te{int(t1/oneday)}_R{R}_{point_loc[0]}E_{point_loc[1]}N'
path_save_png = path_save_png + namesave_loc_area+'/'
path_save_models = path_save_models +namesave_loc_area+'/'
path_save_output = path_save_output +namesave_loc_area+'/'
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
    dict_models['slab_kt_2D'] = classic_slab.jslab_kt_2D(pk, dTK, myforcing, call_args, extra_args)
    dict_models['slab_kt_2D_adv'] = classic_slab.jslab_kt_2D_adv(pk, dTK, myforcing, call_args, extra_args)
    
    # initialize unsteak models
    if Nl==1:
        pk = jnp.asarray([-11.31980127, -10.28525189])    
    elif Nl==2:
        pk = jnp.asarray([-15.,-10., -10., -10.]) 
    NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
    pk = kt_ini(pk, NdT)
    dict_models['unsteak_kt_2D'] = unsteak.junsteak_kt_2D(pk, dTK, myforcing, Nl, call_args, extra_args)   
    dict_models['unsteak_kt_2D_adv2l'] = unsteak.junsteak_kt_2D_adv(pk, dTK, myforcing, Nl, call_args, extra_args, alpha=[1.,1.]) 
    dict_models['unsteak_kt_2D_adv1l'] = unsteak.junsteak_kt_2D_adv(pk, dTK, myforcing, Nl, call_args, extra_args, alpha=[1.,0.])    
    
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
            myt2 = clock.time()
            var = inv.Variational(mymodel, myobservation)
            mymodel, _ = var.scipy_lbfgs_wrapper(mymodel, maxiter, verbose=True)   
            eqx.tree_serialise_leaves(path_save_models+f'{model_name}.pt', mymodel)
            print(f'        time, minimize {model_name} = {clock.time()-myt2}')
        print(f'time, minimize all = {clock.time()-tstart}')

    ################################################
    # STEP 2:
    #
    # Load back the models with their parameters
    #   and save the reconstructed currents as .nc
    ################################################  
    
    
    if SAVE_AS_NC:
        print('* saving each model outputs')
        for model_name, mymodel in dict_models.items():
            dict_models[model_name] = eqx.tree_deserialise_leaves(path_save_models+f'{model_name}.pt', mymodel)
        
        for model_name, mymodel in dict_models.items():
            # model_name = type(model).__name__
            name_save = model_name+'_'+namesave_loc_area
            save_output_as_nc(mymodel, myforcing, LON_bounds, LAT_bounds, name_save, path_save_output)
        print('     done !')
    ################################################
    # STEP 3:
    #
    # Load .nc datasets for analysis, then plots
    ################################################
    location = location_to_plot
    
    datas = {"slab_kt_2D":xr.open_mfdataset(path_save_output+f"slab_kt_2D_{namesave_loc_area}.nc"),
             "slab_kt_2D_adv":xr.open_mfdataset(path_save_output+f"slab_kt_2D_adv_{namesave_loc_area}.nc"),
             "unsteak_kt_2D":xr.open_mfdataset(path_save_output+f"unsteak_kt_2D_{namesave_loc_area}.nc"),
             "unsteak_kt_2D_adv2l":xr.open_mfdataset(path_save_output+f"unsteak_kt_2D_adv2l_{namesave_loc_area}.nc"),
             "unsteak_kt_2D_adv1l":xr.open_mfdataset(path_save_output+f"unsteak_kt_2D_adv1l_{namesave_loc_area}.nc")}
    
    colors = {"slab_kt_2D":'b',
             "slab_kt_2D_adv":'g',
             "unsteak_kt_2D":'orange',
             "unsteak_kt_2D_adv2l":'r',
             "unsteak_kt_2D_adv1l":'pink'}
    
    ls = {"slab_kt_2D":'-',
             "slab_kt_2D_adv":'--',
             "unsteak_kt_2D":'-',
             "unsteak_kt_2D_adv2l":'--',
             "unsteak_kt_2D_adv1l":'--'}
    
    print('* Plotting ...')
    print(f'     saving at: {path_save_png}')
    
    
    # reconstruction at point_loc ----------------
    if PLOT_TRAJ:
        dir = 0 # 0=U, 1=V
        fc = myforcing.fc.mean()
        step_obs = int(period_obs//dt_forcing)
        figsize = (5,5)
        lfontsize = 7
        L_slab = ['slab_kt_2D','slab_kt_2D_adv']
        L_unsteak = ['unsteak_kt_2D','unsteak_kt_2D_adv2l','unsteak_kt_2D_adv1l']
        meta_L = {'slab':L_slab, 'unsteak':L_unsteak}
        
        for metaname, L_models in meta_L.items():
            fig, ax = plt.subplots(2,1,figsize = figsize,constrained_layout=True, dpi=dpi, gridspec_kw={'height_ratios': [3, 1]})
            for model_name in L_models:
                data = datas[model_name]
                Ua = data.Ca.sel(current=dir).sel(lon=location[0],lat=location[1],method='nearest')    
                xtime = np.arange(t0, t1, dt_forcing) / oneday  
                ax[0].plot(xtime, Ua, label=model_name, c=colors[model_name], ls=ls[model_name])
            
            truth = data.C.sel(lon=location[0],lat=location[1],method='nearest').values
            truth_NIO = tools.my_fc_filter(dt_forcing, truth[:,0]+1j*truth[:,1], fc)
            
            ax[0].plot(xtime, truth[:,0], c='k', label='truth',alpha=0.5)
            ax[0].plot(xtime, truth_NIO[0], c='k', label='truth_NIO',alpha=1)
            ax[0].scatter(xtime[::step_obs], truth[::step_obs,0], marker='o',c='r',label='obs')
            ax[0].set_ylabel('zonal current (m/s)')
            ax[0].set_title(f'{metaname}: location is {location}')
            ax[0].legend(loc='lower right', fontsize=lfontsize)
            ax[0].grid()
            ax[0].set_ylim([-0.6,0.6])
            ax[0].set_xlim([t0_plot/oneday, t1_plot/oneday])
            ax[1].plot(xtime, myforcing.data.oceTAUX.sel(lon=location[0],lat=location[1],method='nearest')  , c='g', label=r'$\tau_x$')
            ax[1].plot(xtime, myforcing.data.oceTAUY.sel(lon=location[0],lat=location[1],method='nearest')  , c='orange', label=r'$\tau_y$')
            ax[1].legend(fontsize=lfontsize)
            ax[1].set_ylabel('wind stress (N/m2)')
            ax[1].set_xlabel('time (days)')
            ax[1].grid()
            ax[1].set_ylim([-3,3])
            ax[1].set_xlim([t0_plot/oneday, t1_plot/oneday])
            fig.savefig(path_save_png+f'reconstruction_{metaname}_{location[0]}E_{location[1]}N.png')
            
    # Scores RMSE -------------------------------------
    if PLOT_RMSE:
        print('* Scores RMSE')
        print('')
        print(f'-> at location {location}')
        print(f'    RMSE')
        for model_name, data in datas.items():
            Ca = data.Ca.sel(lon=location[0],lat=location[1],method='nearest').values
            C = data.C.sel(lon=location[0],lat=location[1],method='nearest').values
            value = tools.score_RMSE((Ca[:,0],Ca[:,1]), (C[:,0],C[:,1]))
            print(f'    -> model : {model_name}, {value}')
            
        print(f'    RMSE for truth filtered at fc')
        for model_name, data in datas.items():
            Ca = data.Ca.sel(lon=location[0],lat=location[1],method='nearest').values  
            C = data.C.sel(lon=location[0],lat=location[1],method='nearest').values
            C_nio = tools.my_fc_filter(dt_forcing, C[:,0]+1j*C[:,1], fc)
            value = tools.score_RMSE((Ca[:,0],Ca[:,1]), (C_nio[0],C_nio[1]))
            print(f'    -> model : {model_name}, {value}')
        
        print('')
        print(f'-> over the domain')
        print(f'    RMSE')
        for model_name, data in datas.items():
            Ca = data.Ca.values
            C = data.C.values
            value = tools.score_RMSE((Ca[:,0],Ca[:,1]), (C[:,0],C[:,1]))
            print(f'    -> model : {model_name}, {value}')
            
        print(f'    RMSE for truth filtered at fc')
        for model_name, data in datas.items():
            Ca = data.Ca.values  
            C = data.C.values
            # nested vmap for x and y dimensions
            C_nio = jax.vmap(jax.vmap(tools.my_fc_filter, 
                                        in_axes=(None, 1, None), 
                                        out_axes=1
                                        ),
                                    in_axes=(None, 2, None), 
                                    out_axes=2)(dt_forcing, jnp.asarray(C[:,0]+1j*C[:,1]), fc)
            value = tools.score_RMSE((Ca[:,0],Ca[:,1]), (C_nio[0],C_nio[1]))
            print(f'    -> model : {model_name}, {value}')
        
    # Scores PSD -------------------------------------
    if PLOT_DIAGFREQ:
        print('* Scores PSD')
        NORM_FREQ = False
        skip = 1
        nf = 700
        
        
        for model_name, data in datas.items():
            print('     working on', model_name)
            Ca = data.Ca.values.astype('complex128')
            truth = data.C.values.astype('complex128')
            ff, CWr, ACWr, CWe, ACWe = tools.j_rotary_spectra_2D(1., Ca[:,0], Ca[:,1], truth[:,0], truth[:,1], skip=skip, nf=nf)

            ff, CWr_a, ACWr_a, _, _ = tools.j_rotary_spectra_2D(1., Ca[:,0], Ca[:,1], Ca[:,0], Ca[:,1], skip=skip, nf=nf)

            if NORM_FREQ:
                mean_fc = 2*2*np.pi/86164*np.sin(point_loc[1]*np.pi/180)*onehour/(2*np.pi)
                xtxt = r'f/$f_c$'
            else:
                mean_fc = 1
                xtxt = 'h-1'
            
            fig, axs = plt.subplots(2,1,figsize=(7,6), gridspec_kw={'height_ratios': [4, 1.5]})
            axs[0].loglog(ff/mean_fc,CWr, c='k', label='reference')
            axs[0].loglog(ff/mean_fc, CWr_a, c='g', label='model')
            axs[0].loglog(ff/mean_fc,CWe, c='b', label='error (model - truth)')
            #axs[0].axis([2e-3,2e-1, 1e-4,2e0])
            axs[0].grid('on', which='both')
            axs[1].set_xlabel(xtxt)
            axs[0].legend()
            axs[0].set_ylabel('Clockwise PSD (m2/s2)')
            axs[0].set_ylim([1e-5,2.])
            axs[0].title.set_text(model_name) # +f', RMSE={np.round(RMSE,5)}'

            axs[1].semilogx(ff/mean_fc,(1-CWe/CWr)*100, c='b', label='Reconstruction Score')
            #axs[1].axis([2e-3,2e-1,0,100])
            axs[1].set_ylim([0,100])
            axs[1].grid('on', which='both')
            axs[1].set_ylabel('Scores (%)')
            fig.savefig(path_save_png + model_name + '_diagfreq.png')
            
            
    
    # 2D snapshots -----------------------------------
    if PLOT_SNAPSHOT:
        cmap = 'BrBG_r'
        vmin, vmax = -0.5, 0.5
        indt = 239 # 523
        dir = 0         # 0=U, 1=V
        
        # comparing slabs
        fig, ax = plt.subplots(1,2,figsize = (6.5,4),constrained_layout=True,dpi=dpi)
        # -> no adv
        data = datas['slab_kt_2D']
        U = data.C.isel(time=indt, current=dir)
        Ug = data.Cg.isel(time=indt, current=0)
        Vg = data.Cg.isel(time=indt, current=1)
        Ua = data.Ca.isel(time=indt,current=dir)
        lon = data.lon
        lat = data.lat
        ax[0].pcolormesh(lon, lat, Ua, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[0].set_ylabel('lat')
        ax[0].set_title('slab')
        # -> adv
        Ua = datas['slab_kt_2D_adv'].Ca.isel(time=indt,current=dir)
        ax[1].pcolormesh(lon, lat, Ua, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[1].set_title('slab + adv')
        for axe in ax:
            axe.set_xlabel('lon')
            axe.set_aspect(1.)
        fig.suptitle(f'slab at t={np.round(t0/oneday+indt*dt_forcing/oneday,2)}/365, U ageo')
        fig.savefig(path_save_png+f'snapshot_slab_it{indt}_dir{dir}.png')
        
        # comparings unsteak
        fig, ax = plt.subplots(1,2,figsize = (6.5,4),constrained_layout=True,dpi=dpi)
        # -> no adv
        data = datas['unsteak_kt_2D']
        Ua = data.Ca.isel(time=indt,current=dir)
        lon = data.lon
        lat = data.lat
        ax[0].pcolormesh(lon, lat, Ua, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[0].set_ylabel('lat')
        ax[0].set_title('unsteak')
        # -> adv
        Ua = datas['unsteak_kt_2D_adv2l'].Ca.isel(time=indt,current=dir)
        ax[1].pcolormesh(lon, lat, Ua, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[1].set_title('unsteak + adv2l')
        for axe in ax:
            axe.set_xlabel('lon')
            axe.set_aspect(1.)
        fig.suptitle(f'slab at t={np.round(t0/oneday+indt*dt_forcing/oneday,2)}/365, U ageo')
        fig.savefig(path_save_png+f'snapshot_unsteak_it{indt}_dir{dir}.png')

        # truth ageo and geo    
        dx = 0.1*distance_1deg_equator
        dy = dx
        rot = np.gradient(Vg, dx, axis=-1) - np.gradient(Ug, dy, axis=0)
        fig, ax = plt.subplots(2,1,figsize = (4,7),constrained_layout=True,dpi=dpi)
        im = ax[0].pcolormesh(lon, lat, U, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax[0], aspect=50, pad=0.05)
        im = ax[1].pcolormesh(lon, lat, rot, cmap='seismic', vmin=-6e-5, vmax=6e-5) # , vmin=vmin, vmax=vmax
        plt.colorbar(im, ax=ax[1], aspect=50, pad=0.05)
        for axe in ax:
            axe.set_xlabel('lon')
            axe.set_aspect(1.)
        fig.suptitle(f'truth')
        fig.savefig(path_save_png+f'snapshot_truth_it{indt}_dir{dir}.png')

        
    plt.show()
