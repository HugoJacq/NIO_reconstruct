"""
A class that make a observation object.
"""
import xarray as xr
import numpy as np

from tools import nearest, my_fc_filter

class Observation1D:
    """
    Observations of currents fields for 'Unstek1D'
    
    point_loc   : [LON,LAT] location (°E,°N)
    periode_obs : interval of time between obs (s)
    dt_forcing    : timestep of input OSSE model (s)
    path_file   : path to regridded file with obs/forcing variables 
    """
    def __init__(self, point_loc, periode_obs, dt_forcing, path_file):
        
        # from dataset
        ds = xr.open_mfdataset(path_file)
        indx = nearest(ds.lon.values,point_loc[0])
        indy = nearest(ds.lat.values,point_loc[1])
        self.data = ds.isel(lon=indx,lat=indy)
        
        self.U,self.V = self.data.U.values,self.data.V.values
        self.fc = 2*2*np.pi/86164*np.sin(self.data.lat.values*np.pi/180)
        self.dt_forcing = dt_forcing
        self.obs_period = periode_obs
        self.time_obs = np.arange(0., len(self.data.time)*dt_forcing,periode_obs)

    def get_obs(self):
        """
        OSSE of current from the coupled OA model
        """
        # HERE
        # it would be nice to have the same shape for any model output and the observations.
        # use a fill value ?
        # or see https://stackoverflow.com/questions/71692885/handle-varying-shapes-in-jax-numpy-arrays-jit-compatible
        
        if False:
            U, V = my_fc_filter(self.dt_forcing, self.U+1j*self.V, self.fc )
        else:
            U, V = self.U, self.V
        step_obs = int(self.obs_period)//int(self.dt_forcing)
        print(self.obs_period, int(self.obs_period), self.dt_forcing, int(self.dt_forcing), step_obs)
        self.Uo = U[::step_obs]
        self.Vo = V[::step_obs]
        print(self.U.shape,self.Uo.shape)
        return self.Uo,self.Vo
    

class Observation2D:
    """
    Observations of currents fields for 'Unstek1D', 2D (spatial)
    
    - period_obs : float(s) time interval between observations 
    - dt_forcing : obs comes from OSSE, this is the time step of the OSSE.
    - path_file : file with OSSE data (regridded)
    - LON_bounds : LON min and LON max of zone
    - LAT_bounds : LAT min and LAT max of zone
    """
    def __init__(self, periode_obs, dt_forcing, path_file, LON_bounds, LAT_bounds):
        
        # from dataset for OSSE
        ds = xr.open_mfdataset(path_file)        
        self.data = ds.sel(lon=slice(LON_bounds[0],LON_bounds[1]),lat=slice(LAT_bounds[0],LAT_bounds[1]))
        self.U,self.V = self.data.U.values,self.data.V.values
        self.dt_forcing = dt_forcing
        self.obs_period = periode_obs
        
        self.time_obs = np.arange(0, len(self.data.time)*dt_forcing,periode_obs)

    def get_obs(self):
        """
        OSSE of current from the coupled OA model
        """
        step_obs = int(self.obs_period)//int(self.dt_forcing)
        self.Uo = self.U[::step_obs]
        self.Vo = self.V[::step_obs]
        return self.Uo,self.Vo
    