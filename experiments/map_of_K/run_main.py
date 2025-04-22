"""
This script is meant to plot maps of the parameters from models jslab_kt_2D and junsteak_kt_2D

Is there any correlation between map of K and maps of physical feature (such as gradient of geostrophic)
"""
import numpy as np
import time as clock
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qtagg')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
import os
import xarray as xr
sys.path.insert(0, '../../src')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
import time as clock
import tqdm

# import jax 
#jax.config.update('jax_platform_name', 'cpu')


# THIS NEEDS TO BE MERGED INTO A CLEAN SOLUTION
from Listes_models import *

import forcing
import observations
import inv
from constants import *


from functions import model_instanciation, iter_bounds_mapper, compute_and_save_pk, number_of_tile


# ===========================================================================
# PARAMETERS
# ===========================================================================

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
L_model_to_test             = ['jslab_kt_2D','junsteak_kt_2D'] # L_all jslab_kt_2D junsteak_kt_2D
SAVE_PKs                    = True
SHOW_INFO                   = False                 # if SAVE_PKs
path_save_pk = './pk_save/'

# PLOT
idT = -1                    # which K value to plot (which dT segment)
background_var = 'Ug'       # what contour to plot: Ug, gradMg (norme)
background_time = -1        # date of background data
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
    time0 = clock.time()
    file = []
    for ifile in range(len(name_regrid)):
        file.append(path_regrid+name_regrid[ifile])
        

    dsfull = xr.open_mfdataset(file)
    lon = dsfull.lon.values
    lat = dsfull.lat.values
    print(t0,t1)
    ### WARNINGS
    # warning about t1>length of forcing
    if t1 > len(dsfull.time)*dt_forcing:
        print(f'You chose to run the model for t1={t1//oneday} days but the forcing is available up to t={len(dsfull.time)*dt_forcing//oneday} days\n'
                        +f'I will use t1={len(dsfull.time)*dt_forcing//oneday} days')
        t1 = len(dsfull.time)*dt_forcing
    
    ### END WARNINGS
             
    print('Running: run_main.py')   
    print('**************')
    print('full Croco domain:')
    print(f'        LON min {np.round(np.min(lon),2)}, max {np.round(np.max(lon),2)}')
    print(f'        LAT min {np.round(np.min(lat),2)}, max {np.round(np.max(lat),2)}')
    print(f"if multi layer, nl={args_model['Nl']}")
    print('**************\n')
    Nx, Ny = number_of_tile(R, lon, lat)

    #=====================================================
    if SAVE_PKs:
        # map of tiles
        fig, ax = plt.subplots(1,1,figsize = (10,6),constrained_layout=True,dpi=dpi)
        ax.scatter(np.round(np.min(lon),2),np.round(np.min(lat),2))
        ax.add_patch(matplotlib.patches.Rectangle(
                                        (np.round(np.min(lon),2),np.round(np.min(lat),2)), 
                                        np.round(np.max(lon),2) - np.round(np.min(lon),2), 
                                        np.round(np.max(lat),2) - np.round(np.min(lat),2),
                                        fill=False,
                                        edgecolor='k'
                                        ),
                    )
        ax.set_title(f'tile map for side={2*R}°')    
        ax.set_aspect(1)
        
        # looping on each tiles
        k = 0
        for (point_loc, LON_bounds, LAT_bounds) in tqdm.tqdm(iter_bounds_mapper(R, dx=0.1, lon=lon, lat=lat),total=Nx*Ny,disable=SHOW_INFO):
            
            """
            """
            
            # print some infos
            if SHOW_INFO:
                print(f'k={k}, center of tile is {np.round(point_loc[0],2)}°E,{np.round(point_loc[1],2)}°N')
                print(f'       LON_bounds {np.round(LON_bounds[0],2)} -> {np.round(LON_bounds[1],2)}')
                print(f'       LAT_bounds {np.round(LAT_bounds[0],2)} -> {np.round(LAT_bounds[1],2)}\n')
            tile_infos = k, point_loc, LON_bounds, LAT_bounds
            
            # add rectangle on global figure
            ax.add_patch(matplotlib.patches.Rectangle(
                                        (LON_bounds[0],LAT_bounds[0]), 
                                        LON_bounds[1] - LON_bounds[0], 
                                        LAT_bounds[1] - LAT_bounds[0],
                                        fill=False,
                                        edgecolor='r'
                                        ))
            ax.annotate(str(k), (point_loc[0],point_loc[1]), c='r')
            
            # loop on models
            for model_name in L_model_to_test:
                if model_name not in L_2D_models:
                    raise Exception(f'model {model_name} is not 2D, aborting...')
                
                # forcing
                myforcing = forcing.Forcing2D(dt_forcing, t0, t1, file, LON_bounds, LAT_bounds)
                myobservation = observations.Observation2D(period_obs, t0, t1, dt_forcing, file, LON_bounds, LAT_bounds)

                # model initialization
                mymodel = model_instanciation(model_name, myforcing, args_model, call_args, extra_args)
                var = inv.Variational(mymodel, myobservation, filter_at_fc=False)
            
                # minimize and save pk
                compute_and_save_pk(mymodel, var, mini_args, tile_infos, path_save_pk)
                
                k=k+1
            
        fig.savefig(path_save_png+f'tile_map_side{2*R}.png')
    
    
    
    #=====================================================
    print('* Plotting')
    for model_name in L_model_to_test:
        path_data = 'pk_save/'+model_name+'/'
                
        # getting back the data
        L_pk = []
        for k in range(Nx*Ny):
            L_pk.append(np.load(path_data+f'{k}.npy'))

        NdT, npk = L_pk[0].shape
        pks = np.zeros((Ny, Nx, NdT, npk))
        for kx in range(Nx):
            for ky in range(Ny):
                pks[ky,kx] = L_pk[kx*Ny+ky]
                
        L_infos_loc = []
        minLon,maxLon = 0.,-180.
        minLat,maxLat = 90.,0.
        for line in open(path_data+'link.txt'):
            myline = line.split(',')
            parsed = [float(str) for str in myline]
            point_loc = (parsed[1],parsed[2])
            LON_bounds = (parsed[3],parsed[4])
            LAT_bounds = (parsed[5],parsed[6])
            if parsed[3]<minLon: minLon=parsed[3]
            if parsed[4]>maxLon: maxLon=parsed[4]
            if parsed[5]<minLat: minLat=parsed[5]
            if parsed[6]>maxLat: maxLat=parsed[6]
            L_infos_loc.append([point_loc, LON_bounds, LAT_bounds])
                
        x = np.linspace(minLon,maxLon,Nx+1)
        y = np.linspace(minLat,maxLat,Ny+1)
        
             
        if background_var=='Ug':
            data_bck = dsfull.Ug.isel(time=background_time).values
            levels = [-0.8,-0.6,-0.4,-0.2,
                      0.2,0.4,0.6,0.8]   # m/s
            bck_cmap = 'seismic'
        elif background_var=='gradMg':
            dx,dy = 0.1,0.1
            Mg = np.sqrt(dsfull.Ug.isel(time=background_time).values**2 + dsfull.Vg.isel(time=background_time).values**2)   
            dMgdx = np.gradient(Mg, dx, axis=-1)
            dMgdy = np.gradient(Mg, dy, axis=-2)
            gradMg = np.sqrt(dMgdx**2+dMgdy**2)
            data_bck = gradMg
            levels = [1., 1.5, 2.] # in m/s per degree
            bck_cmap = 'plasma'
            
            
        proj = ccrs.PlateCarree()
        data_crs = ccrs.PlateCarree()
        
        pkmax = pks[:,:,idT,:].max()
        pkmin = pks[:,:,idT,:].min()
        
        for ipk in range(npk):
            fig = plt.figure(figsize=(10, 5),constrained_layout=True,dpi=dpi)
            ax = plt.subplot(111, projection=proj)
            s = ax.pcolormesh(x,y, pks[:,:,idT,ipk], cmap='viridis',vmin=pkmin,vmax=pkmax) #, transform=data_crs) 
            ax.contour(lon,lat,data_bck, transform=data_crs, levels=levels, cmap=bck_cmap)
            plt.colorbar(s,ax=ax)
            # for ky in range(Ny):
            #     for kx in range(Nx):
            #         index = kx*Ny+ky
            #         ax.annotate(f'{np.round(L_infos_loc[index][0][0],2)}'+'\n'+f'{np.round(L_infos_loc[index][0][1],2)}',
            #                     (x[kx],y[ky]))
            ax.coastlines()
            ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k')      # hide values on land
            ax.set_extent([minLon, maxLon, minLat, maxLat], crs=proj)
            gl = ax.gridlines(draw_labels=True, crs=proj)
            gl.top_labels = False
            gl.right_labels = False
            
            ax.set_xlabel('Lon')
            ax.set_ylabel('Lat')
            
            mytime = dsfull.time[background_time].values.astype('datetime64[s]').astype(object)
            if idT==-1:
                idt = NdT-1
            ax.set_title(model_name+f': K{ipk}/{npk-1}, dT{idt}/{NdT-1}, '+mytime.strftime('%H:%M:%S %d-%m-%Y '))
        

            fig.savefig(path_save_png + model_name + f'_K{ipk}_dT{idt}_with_{background_var}.png')
        
        
    
        print(f'total execution time: {clock.time() - time0}')
    plt.show()

   


    