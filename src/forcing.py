"""
A class that make a forcing object
"""

import xarray as xr
import numpy as np

from tools import nearest

class Forcing1D:
    """
    Forcing fields for 'Unstek1D'
    
    point_loc   : [LON,LAT] location of the experiment
    dt_forcing  : time step of forcing
    path_file   : path to regrided dataset with forcing fields (stress) 
    """
    def __init__(self, point_loc, dt_forcing, path_file):
        
        # from dataset
        ds = xr.open_mfdataset(path_file)
        indx = nearest(ds.lon.values,point_loc[0])
        indy = nearest(ds.lat.values,point_loc[1])
        self.data = ds.isel(lon=indx,lat=indy)
        self.U,self.V,self.MLD = self.data.U.values,self.data.V.values,self.data.MLD
        self.TAx,self.TAy = self.data.oceTAUX.values,self.data.oceTAUY.values
        self.fc = 2*2*np.pi/86164*np.sin(self.data.lat.values*np.pi/180)
        self.nt = len(self.data.time)
        self.time = np.arange(0,self.nt*dt_forcing,dt_forcing) 
        self.dt_forcing = dt_forcing
        
class Forcing2D:
    """
    Forcing fields for :
        - 'jUnstek1D_Kt_spatial'
        
    dt_forcing  : time step of forcing
    path_file   : path to regrided dataset with forcing fields (stress) 
    LON_bounds  : [LONmin,LONmax] bounds in lon
    LAT_bounds  : [LATmin,LATmax] bounds in lat
    """
    def __init__(self, dt_forcing, path_file, LON_bounds, LAT_bounds):
        
        # from dataset
        ds = xr.open_mfdataset(path_file)       
        self.data = ds.sel(lon=slice(LON_bounds[0],LON_bounds[1]),lat=slice(LAT_bounds[0],LAT_bounds[1]))
        self.U,self.V,self.MLD = self.data.U.values,self.data.V.values,self.data.MLD
        self.TAx,self.TAy = self.data.oceTAUX.values,self.data.oceTAUY.values
        self.fc = 2*2*np.pi/86164*np.sin(self.data.lat.values*np.pi/180)
        self.nt = len(self.data.time)
        self.time = np.arange(0,self.nt*dt_forcing,dt_forcing)
        self.dt_forcing = dt_forcing