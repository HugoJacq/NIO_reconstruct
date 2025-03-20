import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tools import open_croco_sfx_file, nearest, find_indices_ll, find_indices

def fn_new_name(oldname, new_dx, method):
    return oldname+'_'+str(new_dx)+'deg'+'_'+method

def show_diff(path_in, L_filename, new_dx, method, path_save):
    print('* Visual plot of before/after of last file')
    filename = L_filename[-1]
    new_name = fn_new_name(filename, new_dx, method)
    ds, _ = open_croco_sfx_file(path_in+filename+'.nc', lazy=True, chunks={'time':100})
    ds_out = xr.open_dataset(path_save+new_name+'.nc')
    it = 0
    
    # VERIFICATION
    # -> GLOBAL
    # before
    plt.figure(figsize=(9, 5),dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.pcolormesh(ds.lon_rho,ds.lat_rho,ds['temp'][it])
    ax.coastlines()
    ax.set_title('old SST')
    ax.set_extent([-80, -36, 22, 50], crs=ccrs.PlateCarree())
    # after
    plt.figure(figsize=(9, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.pcolormesh(ds_out.lon,ds_out.lat,ds_out['temp'][it])
    ax.coastlines()
    ax.set_title('new SST')
    ax.set_extent([-80, -36, 22, 50], crs=ccrs.PlateCarree())

    for namevar in ['SSH','MLD','U','V','temp','salt','oceTAUX','oceTAUY','Heat_flx_net','frsh_water_net','SW_rad']:
        plt.figure(figsize=(9, 5),dpi=200)
        ax = plt.axes(projection=ccrs.PlateCarree())
        s = ax.pcolormesh(ds_out.lon,ds_out.lat,ds_out[namevar][it])
        plt.colorbar(s,ax=ax)
        ax.coastlines()
        ax.set_title('new '+namevar)
        ax.set_extent([-80, -36, 22, 50], crs=ccrs.PlateCarree())
    
def show_diff_along_space(path_in, L_filename, new_dx, method, path_save):
    filename = L_filename[-1]
    new_name = fn_new_name(filename, new_dx, method)
    # visual plot at a specific LAT
    LAT = 38. # °N
    LON_min = -60. #°E
    LON_max = -55. #°E
    it = 0
    list_var = ['SSH','temp','U','V'] # ,'temp'
    N_CPU=8 # indice search in //
    
    # here we work on rho-located variables to avoid interpolation
    ds, xgrid = open_croco_sfx_file(path_in+filename+'.nc', lazy=True, chunks={'time':100})
    ds_i = xr.open_dataset(path_save+new_name+'.nc')
    
    # new grid
    indy_n = nearest(ds_i.lat.values,LAT)
    
    # old grid 
    # is lat,lon 2D so we need to find the values at similar location
    olddx = 0.02
    oldlon = ds.lon_rho.values
    oldlat = ds.lat_rho.values
    search_lon = np.arange(LON_min,LON_max,olddx)
    dictvar = {}
    L_points = []

    # find indices in //
    for ik in range(len(search_lon)):
        LON = search_lon[ik]
        L_points.append([LON,LAT])
    L_indices = find_indices_ll(L_points, oldlon, oldlat, N_CPU=N_CPU)
    for namevar in list_var:
        dictvar[namevar] = [ds[namevar][it,indices[0][1],indices[0][0]].values for indices in L_indices] 
    
    
    # plot: slice at LAT
    fig, ax = plt.subplots(len(list_var),1,figsize = (len(list_var)*5,5),constrained_layout=True,dpi=200) 
    for k, namevar in enumerate(list_var):
        ax[k].plot(search_lon, dictvar[namevar],c='k',label='truth')
        if namevar in ['U','V']:
            ax[k].scatter(ds_i.lon,(ds_i[namevar]+ds_i[namevar+'g'])[it,indy_n,:],c='b',label='interp',marker='x')
        else:
            ax[k].scatter(ds_i.lon,ds_i[namevar][it,indy_n,:],c='b',label='interp',marker='x')
        ax[k].set_xlim([LON_min,LON_max])
        ax[k].set_ylabel(namevar)
    ax[-1].set_xlabel('LON')
    
def show_diff_along_time(path_in, L_filename, new_dx, method, path_save):
    filename = L_filename[-1]
    
    new_name = fn_new_name(filename, new_dx, method)
    # visual plot at a specific LAT
    LAT = 38. # °N
    select_LON = -60. #°E
    LON_max = -55. #°E
    it = 0
    list_var = ['SSH','temp','U','V'] # ,'temp'
    
    # here we work on rho-located variables to avoid interpolation
    ds, xgrid = open_croco_sfx_file(path_in+filename+'.nc', lazy=True, chunks={'time':100})
    ds_i = xr.open_dataset(path_save+new_name+'.nc')
    time = ds_i.time
    
    # new grid
    indy_n = nearest(ds_i.lat.values,LAT)
    
    # old grid 
    # is lat,lon 2D so we need to find the values at similar location
    olddx = 0.02
    oldlon = ds.lon_rho.values
    oldlat = ds.lat_rho.values

    # plot: slice at LAT,LON_max
    indx,indy = find_indices([select_LON,LAT],oldlon,oldlat)[0]
    indx_n = nearest(ds_i.lon.values,select_LON)
    fig, ax = plt.subplots(len(list_var),1,figsize = (len(list_var)*5,5),constrained_layout=True,dpi=200) 
    for k, namevar in enumerate(list_var):
        ax[k].plot(time, ds[namevar][:,indy,indx],c='k',label='truth')
        #ax[k].scatter(time, ds_i[namevar][:,indy_n,indx_n],c='b',label='interp',marker='x')
        if namevar in ['U','V']:
            ax[k].scatter(time, (ds_i[namevar]+ds_i[namevar+'g'])[:,indy_n,indx_n],c='b',label='interp',marker='x')
        else:
            ax[k].scatter(time, ds_i[namevar][:,indy_n,indx_n],c='b',label='interp',marker='x')
        ax[k].set_ylabel(namevar)
    ax[-1].set_xlabel('time')
    ax[0].set_title('at LON,LAT='+str(select_LON)+','+str(LAT))
    
def show_bili_vs_cons(L_filename, new_dx, path_save):
    
    filename = L_filename[-1]
    new_name_cons = fn_new_name(filename, new_dx, method='conservative')
    new_name_bili = fn_new_name(filename, new_dx, method='bilinear')
    
    # visual plot at a specific LAT
    it = 0
    list_var = ['SSH','temp','U','V'] # ,'temp'

    dsc = xr.open_mfdataset(path_save+new_name_cons+'.nc')
    dsb = xr.open_mfdataset(path_save+new_name_bili+'.nc')
    
    for k, namevar in enumerate(list_var):
        fig, ax = plt.subplots(1,1,figsize = (7,5),constrained_layout=True,dpi=200) 
        s = ax.pcolormesh(dsc.lon,dsc.lat, dsb[namevar][it,:-1,:]-dsc[namevar][it,:,:],cmap='bwr')
        plt.colorbar(s,ax=ax)
        ax.set_ylabel('lat')
        ax.set_xlabel('lon')
        ax.set_title(namevar+' bili minus cons')