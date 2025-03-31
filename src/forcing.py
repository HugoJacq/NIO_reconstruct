"""
A class that make a forcing object
"""

import xarray as xr
import numpy as np
# import cdflib.xarray as xrcdf  # to read cdf files
# import cdflib
 
from tools import nearest, open_PAPA_station_file

class Forcing1D:
    """
    Forcing fields for 'Unstek1D' 1D models
    
    point_loc   : [LON,LAT] location of the experiment
    dt_forcing  : time step of forcing
    path_file   : path to regrided dataset with forcing fields (stress) 
    """
    def __init__(self, point_loc, t0, t1, dt_forcing, path_file):
        
        # from dataset
        ds = xr.open_mfdataset(path_file)
        indx = nearest(ds.lon.values,point_loc[0])
        indy = nearest(ds.lat.values,point_loc[1])
        time_a = np.arange(t0,t1+dt_forcing,dt_forcing)
        itmin = nearest(time_a, t0)
        itmax = nearest(time_a, t1)
        self.data = ds.isel(time=slice(itmin,itmax), lon=indx,lat=indy)
        self.U,self.V,self.MLD = self.data.U.values,self.data.V.values,self.data.MLD
        self.TAx,self.TAy = self.data.oceTAUX.values,self.data.oceTAUY.values
        self.fc = 2*2*np.pi/86164*np.sin(self.data.lat.values*np.pi/180)
        self.nt = len(self.data.time)
        self.time = np.arange(0,self.nt*dt_forcing,dt_forcing) 
        self.dt_forcing = dt_forcing
        
class Forcing2D:
    """
    Forcing fields for :
        - 'jUnstek1D_Kt_spatial' 2D models
        
    dt_forcing  : time step of forcing
    path_file   : path to regrided dataset with forcing fields (stress) 
    LON_bounds  : [LONmin,LONmax] bounds in lon
    LAT_bounds  : [LATmin,LATmax] bounds in lat
    """
    def __init__(self, dt_forcing, t0, t1, path_file, LON_bounds, LAT_bounds):
        
        # from dataset
        ds = xr.open_mfdataset(path_file)  
        time_a = np.arange(t0,t1+dt_forcing,dt_forcing)
        itmin = nearest(time_a, t0)
        itmax = nearest(time_a, t1)
        ds = ds.isel(time=slice(itmin,itmax))     
        self.data = ds.sel(lon=slice(LON_bounds[0],LON_bounds[1]),lat=slice(LAT_bounds[0],LAT_bounds[1]))
        self.U,self.V,self.MLD = self.data.U.values,self.data.V.values,self.data.MLD
        self.Ug, self.Vg = self.data.Ug.values, self.data.Vg.values
        self.TAx,self.TAy = self.data.oceTAUX.values,self.data.oceTAUY.values
        self.fc = 2*2*np.pi/86164*np.sin(self.data.lat.values*np.pi/180)
        self.nt = len(self.data.time)
        self.time = np.arange(0,self.nt*dt_forcing,dt_forcing)
        self.dt_forcing = dt_forcing
        
class Forcing_from_PAPA:
    """
    Forcing fields from the PAPA sation, located at 50.1N 144.9W    
    """
    
    def __init__(self, dt_forcing, t0, t1, path_file):
        
        LAT = 50.1 # °N        
        
        # opening dataset
        ds = open_PAPA_station_file(path_file)        
        time_a = np.arange(t0,t1+dt_forcing,dt_forcing)
        itmin = nearest(time_a, t0)
        itmax = nearest(time_a, t1)
        ds = ds.isel(time=slice(itmin,itmax))     
        self.data = ds.isel(time=slice(itmin,itmax))
        self.U,self.V = self.data.U.values,self.data.V.values
        self.TAx,self.TAy = self.data.TAx.values,self.data.TAy.values # stress = Cd*U**2 !
        self.fc = 2*2*np.pi/86164*np.sin(LAT*np.pi/180)
        self.nt = len(self.data.time)
        self.time = np.arange(0,self.nt*dt_forcing,dt_forcing) 
        self.dt_forcing = dt_forcing