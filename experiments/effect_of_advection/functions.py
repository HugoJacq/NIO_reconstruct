import time as clock
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../../src')
import os
import xarray as xr

import tools
import models.classic_slab as classic_slab
import models.unsteak as unsteak
from basis import kt_ini, kt_1D_to_2D, pkt2Kt_matrix
from constants import *
from inv import my_partition
from Listes_models import L_slabs, L_unsteaks, L_variable_Kt, L_nlayers_models, L_models_total_current


def save_output_as_nc(mymodel, forcing2D, LON_bounds, LAT_bounds, name_save, where_to_save):
    
    # getting data
    Ua,Va = mymodel(save_traj_at=mymodel.dt_forcing)
    if type(mymodel).__name__ in L_nlayers_models:
        Ua, Va = Ua[:,0], Va[:,0]
    U = forcing2D.data.U
    V = forcing2D.data.V
    Ug = forcing2D.data.Ug
    Vg = forcing2D.data.Vg
    
    t0, t1 = mymodel.t0, mymodel.t1
    x = np.arange(LON_bounds[0],LON_bounds[1], 0.1)
    y = np.arange(LAT_bounds[0],LAT_bounds[1], 0.1)  
    
    #time = np.arange(t0,t1,forcing2D.dt_forcing)
    
    """
    TO BE ADDED
    - the pk vector
    - has the parameters been through minimization ?
    - U truth
    - model static parameters: time window, dt, ...
    - origin of truth -> Croco
    
    - let the possibility for the user to add a note in the terminal window ??
    
    """
    
    os.system('mkdir -p ' + where_to_save)
    
    # making the dataset
    ds = xr.Dataset({"Ua": (["time", "lat", "lon"], np.asarray(Ua)),
                     "Ug": (["time", "lat", "lon"], Ug.data),
                     "Uat": (["time", "lat", "lon"], np.asarray(Ua)+Ug.values),
                     "Ut": (["time", "lat", "lon"], (U+Ug).data),
                    },
                    coords={
                        "lon": (["lon"], x),
                        "lat": (["lat"], y),
                        "time": forcing2D.data.time,
                            },
                    attrs={
                        "made_from":"save_output_as_nc",
                        "model_used":type(mymodel).__name__
                    }

                    )
    ds.to_netcdf(where_to_save + name_save + '.nc')