import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from matplotlib.patches import Rectangle

# INPUTS
namefile = 'croco_1h_inst_surf_2005-07-01-2005-07-31'
file_path_HR = '/ODYSEA2/Lionel_coupled_run/' + namefile + '.nc'
file_path_LR = '../data_regrid/' + namefile + '_0.1deg_conservative.nc'
vmin=-1
vmax=1
cmap=cmocean.cm.deep_r
idt = -1
proj = ccrs.PlateCarree()
data_crs = ccrs.PlateCarree()
point_loc = [-47.4,34.6] 
R = 5 # °
# opening files

dsLR = xr.open_dataset(file_path_LR)
dsLR['Ut'] = dsLR['Ug']+dsLR['U']


lonmin,lonmax = np.min(dsLR.lon), np.max(dsLR.lon)
latmin,latmax = np.min(dsLR.lat), np.max(dsLR.lat)

if False:
    dsHR = xr.open_dataset(file_path_HR)
    fig = plt.figure(figsize=(10, 3),constrained_layout=True, dpi=200)
    ax1 = plt.subplot(121, projection=proj)
    ax1.pcolormesh(dsHR.nav_lon_rho, dsHR.nav_lat_rho, dsHR['u'][idt], cmap=cmap, vmin=vmin, vmax=vmax)
    ax2 = plt.subplot(122, projection=proj)
    s = ax2.pcolormesh(dsLR.lon, dsLR.lat, dsLR['Ut'][idt], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(s,ax=ax2, aspect=50)



    for axe in [ax1,ax2]:
        axe.coastlines()
        axe.add_feature(cfeature.LAND, zorder=1, edgecolor='k')      # hide values on land
        axe.set_extent([lonmin, lonmax, latmin, latmax], crs=proj)
        gl = axe.gridlines(draw_labels=True, crs=proj, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
    gl.left_labels = False

    fig.savefig('nice_plot_Ut_2maps.png')

if True:
    fig = plt.figure(figsize=(6, 3),constrained_layout=True, dpi=200)
    ax1 = plt.subplot(111, projection=proj)
    s = ax1.pcolormesh(dsLR.lon, dsLR.lat, dsLR['Ut'][idt], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(s,ax=ax1, aspect=50, pad=0.05)
    ax1.coastlines()
    ax1.add_feature(cfeature.LAND, zorder=1, edgecolor='k')      # hide values on land
    ax1.set_extent([lonmin, lonmax, latmin, latmax], crs=proj)
    gl = ax1.gridlines(draw_labels=True, crs=proj, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    ax1.scatter(point_loc[0],point_loc[1],marker='x',c='r')
    ax1.add_patch(Rectangle((point_loc[0]-R, point_loc[1]-R), 2*R, 2*R, edgecolor = 'red',facecolor = 'blue', fill=False,))
    fig.savefig('nice_plot_Ut_LR.png')
    
plt.show()