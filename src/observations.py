"""
A class that make a observation object.
"""
import xarray as xr
import numpy as np

from tools import nearest, my_fc_filter, open_PAPA_station_file
from constants import distance_1deg_equator

class Observation1D:
    """
    Observations of currents fields for 'Unstek1D'
    
    point_loc   : [LON,LAT] location (°E,°N)
    periode_obs : interval of time between obs (s)
    dt_forcing    : timestep of input OSSE model (s)
    path_file   : path to regridded file with obs/forcing variables 
    """
    def __init__(self, point_loc, periode_obs, t0, t1, dt_forcing, path_file):
        
        # from dataset
        ds = xr.open_mfdataset(path_file)
        indx = nearest(ds.lon.values,point_loc[0])
        indy = nearest(ds.lat.values,point_loc[1])
        time_a = np.arange(0,len(ds.time)*dt_forcing, dt_forcing)
        itmin = nearest(time_a, t0)
        itmax = nearest(time_a, t1)
        self.data = ds.isel(time=slice(itmin,itmax), lon=indx,lat=indy)
        
        self.U,self.V = self.data.U.values,self.data.V.values
        self.Ug,self.Vg = self.data.Ug.values,self.data.Vg.values
        self.fc = 2*2*np.pi/86164*np.sin(self.data.lat.values*np.pi/180)
        self.dt_forcing = dt_forcing
        self.obs_period = periode_obs
        self.time_obs = np.arange(0., len(self.data.time)*dt_forcing,periode_obs)

    def get_obs(self, is_utotal=False):
        """
        OSSE of current from the coupled OA model
        """
        # HERE
        # it would be nice to have the same shape for any model output and the observations.
        # use a fill value ?
        # or see https://stackoverflow.com/questions/71692885/handle-varying-shapes-in-jax-numpy-arrays-jit-compatible
        
        

        if is_utotal:
            U, V = self.U + self.Ug , self.V + self.Vg
        else:
            U, V = self.U, self.V
            
        if False:
            U, V = my_fc_filter(self.dt_forcing, U+1j*V, self.fc )
        
        step_obs = int(self.obs_period)//int(self.dt_forcing)
        self.Uo = U[::step_obs]
        self.Vo = V[::step_obs]
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
    def __init__(self, periode_obs, t0, t1, dt_forcing, path_file, LON_bounds, LAT_bounds):
        
        # from dataset for OSSE
        ds = xr.open_mfdataset(path_file) 
        time_a = np.arange(0,len(ds.time)*dt_forcing, dt_forcing)
        itmin = nearest(time_a, t0)
        itmax = nearest(time_a, t1)
        ds = ds.isel(time=slice(itmin,itmax))        
        self.data = ds.sel(lon=slice(LON_bounds[0],LON_bounds[1]),lat=slice(LAT_bounds[0],LAT_bounds[1]))
        self.U,self.V = self.data.U.values,self.data.V.values
        self.Ug,self.Vg = self.data.Ug.values,self.data.Vg.values
        self.dt_forcing = dt_forcing
        self.obs_period = periode_obs
        self.dx_deg,self.dy_deg = 0.1, 0.1
        self.dx,self.dy = self.dx_deg*distance_1deg_equator, self.dy_deg*distance_1deg_equator
        
        self.time_obs = np.arange(0, len(self.data.time)*dt_forcing,periode_obs)

    def get_obs(self, is_utotal=False):
        """
        OSSE of current from the coupled OA model
        """
        if is_utotal:
            U, V = self.U + self.Ug , self.V + self.Vg
        else:
            U, V = self.U, self.V
        step_obs = int(self.obs_period)//int(self.dt_forcing)
        self.Uo = U[::step_obs]
        self.Vo = V[::step_obs]
        return self.Uo,self.Vo
    
class Observation_from_PAPA:
    """
    Observations from the station PAPA
    """
    
    def __init__(self, periode_obs, t0, t1, dt_forcing, path_file):
        
        # opening dataset
        ds = open_PAPA_station_file(path_file)        
        time_a = np.arange(0,len(ds.time)*dt_forcing, dt_forcing)
        itmin = nearest(time_a, t0)
        itmax = nearest(time_a, t1)    
        self.data = ds.isel(time=slice(itmin,itmax))
        self.U,self.V = self.data.U.values,self.data.V.values
        self.dt_forcing = dt_forcing
        self.obs_period = periode_obs
        # self.time_obs = np.arange(t0, t1, periode_obs)
        self.time_obs = np.arange(0, len(self.data.time)*dt_forcing,periode_obs)
        
    def get_obs(self, is_utotal=False):
        """
        Get current time spaced by 'obs_period'
        
        Here, we make the hypothesis Utotal = Uag
        """
        step_obs = int(self.obs_period)//int(self.dt_forcing)
        self.Uo = self.U[::step_obs]
        self.Vo = self.V[::step_obs]
        return self.Uo,self.Vo
    
class Observations_idealized_1D:
    def __init__(self, periode_obs, dt_forcing, t0, t1):
        
        time = np.arange(0, t1-t0, dt_forcing) 
        TAx = 0.4
        TAy = 0.
        self.TAx,self.TAy = np.ones(len(time))*TAx, np.zeros(len(time))
        
        # impulse response
        r=5e-6
        K0 = 1e-4
        f = 0.0001
        C = K0 * 1/(r+1j*f) * ( 1-np.exp(- ((r+1j*f)*time)) ) * (TAx+1j*TAy)
        self.U,self.V = np.real(C), np.imag(C)
        self.time = np.arange(0, len(time)*dt_forcing,periode_obs)
        self.dt_forcing = dt_forcing
        self.obs_period = periode_obs

    def get_obs(self):
        """
        Get current time spaced by 'obs_period'
        """
        step_obs = int(self.obs_period)//int(self.dt_forcing)
        self.Uo = self.U[::step_obs]
        self.Vo = self.V[::step_obs]
        return self.Uo,self.Vo
    
class Observations_idealized_2D:
    def __init__(self, periode_obs, dt_forcing, t0, t1, nx, ny):
        
        time = np.arange(0, t1-t0, dt_forcing) 
        TAx = 0.4
        TAy = 0.

        ID =  np.ones((len(time), ny, nx))
        self.TAx,self.TAy = ID*TAx, ID*TAy
        
        # impulse response
        r=5e-6
        K0 = 1e-4
        f = 0.0001
        C = ( K0 * 1/(r+1j*f) * ( 1-np.exp(- ((r+1j*f)*time)) ) * (self.TAx+1j*self.TAy).T ).T
        self.U,self.V = np.real(C), np.imag(C)
        self.time = np.arange(0, len(time)*dt_forcing,periode_obs)
        self.dt_forcing = dt_forcing
        self.obs_period = periode_obs

    def get_obs(self):
        """
        Get current time spaced by 'obs_period'
        """
        step_obs = int(self.obs_period)//int(self.dt_forcing)
        self.Uo = self.U[::step_obs]
        self.Vo = self.V[::step_obs]
        return self.Uo,self.Vo