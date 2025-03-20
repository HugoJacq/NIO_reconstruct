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
        ds = xr.open_dataset(path_file)
        indx = nearest(ds.lon.values,point_loc[0])
        indy = nearest(ds.lat.values,point_loc[1])
        self.data = ds.isel(lon=indx,lat=indy)
        
        self.U,self.V,self.MLD = self.data.U.values,self.data.V.values,self.data.MLD
        # if 'TAx' in self.data.keys():
        #     self.bulkTx,self.bulkTy = self.data.TAx.values,self.data.TAy.values
        self.TAx,self.TAy = self.data.oceTAUX.values,self.data.oceTAUY.values
        self.fc = 2*2*np.pi/86164*np.sin(self.data.lat.values*np.pi/180) # Coriolis value at jr,ir
        self.nt = len(self.data.time)
        self.time = np.arange(0,self.nt*dt_forcing,dt_forcing)    # 1 step every dt
        self.dt_forcing = dt_forcing