"""
Function that perform regriding of Croco output
"""

import sys
import os
#sys.path.append('..')
sys.path.insert(0, '../src')
import xarray as xr
import numpy as np
import xesmf as xe
import time as clock
import pathlib

from tools import open_croco_sfx_file
from filters import mytimefilter_over_spatialXY
from constants import *

def regridder(path_file, namefile, method, new_dx, path_save, N_CPU=8):
    """
    """ 
    start = clock.time()
    print('')
    print(' -> Regridding:', path_file)
    print('')
    new_name = namefile+'_'+str(new_dx)+'deg'+'_'+method
    if pathlib.Path(path_save+new_name+'.nc').is_file():
        print(' -> File is here !')
        print('     '+path_save+new_name+'.nc')
        
    else: 

        print('     * Opening file ...')
        ds, xgrid = open_croco_sfx_file(path_file+namefile+'.nc', lazy=True, chunks={'time':100})

        print('     * Getting land mask ...')
        ds['mask_valid'] = xr.where(~(np.isfinite(ds.SSH)),0.,1.).astype(np.float32).compute()

        # fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        # s=ax.pcolormesh(ds['mask_valid'].isel(time=0),cmap='jet')
        # plt.colorbar(s,ax=ax)
        # ax.set_xlabel('lon')
        # ax.set_ylabel('lat')
        # ax.set_title('mask_valid')
        # plt.show()

        print('     * Interpolation at mass point ...')
        # Croco is grid-C
        L_u = ['U','oceTAUX']
        L_v = ['V','oceTAUY']
        for var in L_u:
            attrs = ds[var].attrs
            ds[var] = xgrid.interp(ds[var], 'x') # .load()
            ds[var].attrs = attrs
        for var in L_v:
            attrs = ds[var].attrs
            ds[var] = xgrid.interp(ds[var], 'y') # .load()
            ds[var].attrs = attrs
                
        # we have variables only at rho points now
        #   so we rename the coordinates with names
        #   that xesmf understand
        ds = ds.rename({"lon_rho": "lon", "lat_rho": "lat"})
        ds = ds.set_coords(['lon','lat'])

        if method=='conservative':  
            # we need to compute the boundaries of the lat,lon.
            # here we are high res so we just remove 1 data point.
            # see https://earth-env-data-science.github.io/lectures/models/regridding.html
            ds['lon_b'] = xr.DataArray(ds.lon_u.data[1:,:], dims=["y_u", "x_u"])
            ds['lat_b'] = xr.DataArray(ds.lat_v.data[:,1:] , dims=["y_v", "x_v"])            
            ds['lon_b'] = ds['lon_b'].where(ds['mask_valid'].values[0,1:,1:])#ds.lon_b==0.,np.nan,ds.lon_b)
            ds['lat_b'] = ds['lat_b'].where(ds['mask_valid'].values[0,1:,1:])#ds.lat_b==0.,np.nan,ds.lat_b)    
            ds = ds.set_coords(['lon_b','lat_b'])
            ds = ds.isel(x_rho=slice(0,-2),y_rho=slice(0,-2))

        # remove unused coordinates
        ds = ds.reset_coords(['lon_v','lat_v','lon_u','lat_u'],drop=True)

        # mask area where lat and lon == 0.
        lon2D = ds.lon
        lat2D = ds.lat
        lonmin = np.round( np.nanmin(np.where(lon2D.values<0.,lon2D.values,np.nan)), 1)
        lonmax = np.round( np.nanmax(np.where(lon2D.values<0.,lon2D.values,np.nan)), 1)
        latmin = np.round( np.nanmin(np.where(lat2D.values>0.,lat2D.values,np.nan)), 1)
        latmax = np.round( np.nanmax(np.where(lat2D.values>0.,lat2D.values,np.nan)), 1)
        print('     * Area is:')
        print('         LON = [', lonmin,lonmax,']')
        print('         LAT = [', latmin,latmax,']')

        ds['lon'] = ds['lon'].where(ds['mask_valid'].values[0])#ds.lon==0.,np.nan,ds.lon)
        ds['lat'] = ds['lat'].where(ds['mask_valid'].values[0])#ds.lat==0.,np.nan,ds.lat)

        #ds['mask_valid'] = xr.where(lon2D==0.,0.,1.)

        print('     * Regridding ...')
        # new dataset
        ds_out = xe.util.grid_2d(lonmin, lonmax, new_dx, latmin, latmax, new_dx)       

        # regridder
        print('         -> compute once the operator')
        regridder = xe.Regridder(ds, ds_out, method) # bilinear conservative

        # regriding variables+
        # some error is raised about arrays not being C-contiguous, please ignore
        print('         -> apply the operator to variables:')
        ds_out['mask_valid'] = regridder(ds['mask_valid'].where(ds['mask_valid']==1))
        for namevar in list(ds.variables):
            if namevar not in ['lat', 'lon', 'lat_u', 'lon_u', 'lat_v', 'lon_v', 'time','mask_valid','lon_b','lat_b']:
                print('     '+namevar)
                attrs = ds[namevar].attrs
                ds_out[namevar] = regridder(ds[namevar])
                ds_out[namevar].attrs = attrs
                # masking
                ds_out[namevar] = ds_out[namevar].where(ds_out['mask_valid'])

        # print(ds_out.U)
        # print(ds_out.U.values)


        # replacing x and y with lon1D and lat1D
        ds_out['lon1D'] = ds_out.lon[0,:]
        ds_out['lat1D'] = ds_out.lat[:,0]
        ds_out = ds_out.swap_dims({'x':'lon1D','y':'lat1D'})

        # removing unsed dims and coordinates
        ds_out = ds_out.drop_dims(['x_b','y_b'])
        ds_out = ds_out.reset_coords(names=['lon','lat'], drop=True)
        ds_out = ds_out.rename({'lon1D':'lon','lat1D':'lat'})


        # COMPUTING AND REMOVING GEOSTROPHY ---
        # -> large scale filtering of SSH
        print('     * Computing geostrophy ...')

        # preparing fields
        ds_out['SSH'] = xr.where(ds_out['SSH']>1e5,np.nan,ds_out['SSH']).compute() # this trigger the interpolation process from previous steps
        ds_out['SSH_LS0'] = xr.zeros_like(ds_out['SSH'])
        ds_out['SSH_LS'] = xr.zeros_like(ds_out['SSH'])

        # -> smoothing: spatial
        print('         2D filter')
        # spatial filter is not done here
        # because SSH is already smoothed by redrid AND the following time filter (moving structures)
        # if the regrid file resolution is closer to the original resolution, you might need to add a spatial filter !
        ds_out['SSH_LS0'].data = ds_out['SSH'].data

        # -> smoothing: time
        print('         time filter')
        ds_out['SSH_LS'].data = mytimefilter_over_spatialXY(ds_out['SSH_LS0'].values, N_CPU=N_CPU, show_progress=True) 

        # mask invalid data
        #   and load
        ds_out['mask_valid'] = xr.where(~(np.isfinite(ds_out['mask_valid'])),0.,ds_out['mask_valid']).astype(np.float32).compute()
        ds_out['SSH_LS'] = ds_out['SSH_LS'].where(ds_out.mask_valid.data).compute()

        # fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        # s=ax.pcolormesh(ds_out['mask_valid'].isel(time=0),cmap='jet')
        # plt.colorbar(s,ax=ax)
        # ax.set_xlabel('lon')
        # ax.set_ylabel('lat')
        # ax.set_title('mask_valid ds_out')
        # plt.show()

        # -> getting geo current from SSH
        print('         gradXY ssh')
        glon2,glat2 = np.meshgrid(ds_out['lon'].values,ds_out['lat'].values)

        # glat2 = ds_out['lat'].values
        # glon2 = ds_out['lon'].values
        dlon = (ds_out['lon'][1]-ds_out['lon'][0]).values
        dlat = (ds_out['lat'][1]-ds_out['lat'][0]).values
        fc = 2*2*np.pi/86164*np.sin(glat2*np.pi/180)
        gUg = ds_out['SSH_LS']*0.
        gVg = ds_out['SSH_LS']*0.


        # gradient metrics
        dx = dlon * np.cos(np.deg2rad(glat2)) * distance_1deg_equator 
        dy = ((glon2 * 0) + 1) * dlat * distance_1deg_equator
                
        # over time array
        for it in range(len(ds_out.time)):
            # this could be vectorized ...
            # centered gradient, dSSH/dx is on SSH point
            gVg[it,:,1:-1] =  grav/fc[:,1:-1]*( ds_out['SSH_LS'][it,:,2:].values - ds_out['SSH_LS'][it,:,:-2].values ) / ( dx[:,1:-1] )/2
            gUg[it,1:-1,:] = -grav/fc[1:-1,:]*(  ds_out['SSH_LS'][it,2:,:].values  - ds_out['SSH_LS'][it,:-2,:].values ) / ( dy[1:-1,:] )/2
        gUg[:,0,:] = gUg[:,1,:]
        gUg[:,-1,:]= gUg[:,-2,:]
        gVg[:,:,0] = gVg[:,:,1]
        gVg[:,:,-1]= gVg[:,:,-2]

        # adding geo current to the file
        ds_out['Ug'] = (ds_out['SSH'].dims,
                            gUg.data,
                            {'standard_name':'zonal_geostrophic_current',
                                'long_name':'zonal geostrophic current from SSH',
                                'units':'m s-1',})
        ds_out['Vg'] = (ds_out['SSH'].dims,
                            gVg.data,
                            {'standard_name':'meridional_geostrophic_current',
                                'long_name':'meridional geostrophic current from SSH',
                                'units':'m s-1',}) 

        # removing geostrophy
        ds_out['U'].data = ds_out['U'].values - ds_out['Ug'].values
        ds_out['V'].data = ds_out['V'].values - ds_out['Vg'].values
        ds_out['U'].attrs['long_name'] = 'Ageo '+ds_out['U'].attrs['long_name']
        ds_out['V'].attrs['long_name'] = 'Ageo '+ds_out['V'].attrs['long_name']   

        # fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        # s=ax.pcolormesh(ds_out['Ug'].isel(time=0),cmap='jet',vmin=-1,vmax=1)
        # plt.colorbar(s,ax=ax)
        # ax.set_xlabel('lon')
        # ax.set_ylabel('lat')
        # ax.set_title('Ug ds_out')
                
        # fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        # s=ax.pcolormesh((ds_out['Ug']+ds_out['U']).isel(time=0),cmap='jet',vmin=-1,vmax=1)
        # plt.colorbar(s,ax=ax)
        # ax.set_xlabel('lon')
        # ax.set_ylabel('lat')
        # ax.set_title('Ug+Uageo ds_out')

        # fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        # s=ax.pcolormesh(ds['U'].isel(time=0),cmap='jet',vmin=-1,vmax=1)
        # plt.colorbar(s,ax=ax)
        # ax.set_xlabel('lon')
        # ax.set_ylabel('lat')
        # ax.set_title('Ug+Uageo ds')

        # fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        # s=ax.pcolormesh(ds_out['Vg'].isel(time=0),cmap='jet',vmin=-1,vmax=1)
        # plt.colorbar(s,ax=ax)
        # ax.set_xlabel('lon')
        # ax.set_ylabel('lat')
        # ax.set_title('Vg ds_out')

        # fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        # s=ax.pcolormesh((ds_out['Vg']+ds_out['V']).isel(time=0),cmap='jet',vmin=-1,vmax=1)
        # plt.colorbar(s,ax=ax)
        # ax.set_xlabel('lon')
        # ax.set_ylabel('lat')
        # ax.set_title('Vg+Vageo ds_out')

        # fig, ax = plt.subplots(1,1,figsize = (10,5), constrained_layout=True,dpi=200)
        # s=ax.pcolormesh(ds['V'].isel(time=0),cmap='jet',vmin=-1,vmax=1)
        # plt.colorbar(s,ax=ax)
        # ax.set_xlabel('lon')
        # ax.set_ylabel('lat')
        # ax.set_title('Vg+Vageo ds')

        # plt.show()

        # removing work variables
        ds_out = ds_out.drop_vars(['SSH_LS0','SSH_LS','SW_rad'])
        # END COMPUTING GEOSTROPHY ---



        # print some stats
        print('\n   OLD DATASET\n')
        print(ds)
        print('\n   NEW DATASET\n')
        print(ds_out)

        print('     * Saving ...')
        ds_out.attrs['xesmf_method'] = method
        ds_out.compute()
        ds_out.to_netcdf(path=new_name + '.nc',mode='w')
        ds.close()
        ds_out.close()

        # save regridder
        regridder.to_netcdf(path_save+'regridder_'+str(new_dx)+'deg_'+method+'.nc')

        end = clock.time()
        print('     Execution time for this file = '+str(np.round(end-start,2))+' s')
    
def mf_regridder(path_file, namefile, method, new_dx, path_save, N_CPU=8):
    
    if len(namefile)==1:
        regridder(path_file, namefile[0], method, new_dx, path_save,N_CPU)
    else:
        for ifile in range(len(namefile)):
            regridder(path_file, namefile[ifile], method, new_dx, path_save, N_CPU)