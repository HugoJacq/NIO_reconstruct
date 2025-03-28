"""
Here we want to find the optimal dTK for the vector pk.
"""


import numpy as np
import time as clock
import matplotlib.pyplot as plt
import sys
import os
import xarray as xr
sys.path.insert(0, '../../src')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
import jax
import jax.numpy as jnp

from models.classic_slab import jslab, jslab_kt, jslab_kt_2D, kt_ini, kt_1D_to_2D, pkt2Kt_matrix
import forcing
import inv
import observations
#from tests_functions import run_forward_cost_grad, plot_traj_1D, plot_traj_2D
import tools
from constants import *

start = clock.time()



# ============================================================
# PARAMETERS
# ============================================================
#ON_HPC      = False      # on HPC

# model parameters
Nl                  = 1         # number of layers for multilayer models
dTK                 = 10*oneday   # how much vectork K changes with time, basis change to exp
k_base              = 'gauss'   # base of K transform. 'gauss' or 'id'
AD_mode             = 'F'       # forward mode for AD 

# run parameters
t0                  = 0*oneday
t1                  = 365*oneday
dt                  = 60.        # timestep of the model (s) 

# What to test
FORWARD_PASS        = False      # tests forward, cost, gradcost
MINIMIZE            = True      # switch to do the minimisation process
maxiter             = 100         # max number of iteration
PLOT_TRAJ           = True



TEST_SLAB_KT                = True

# PLOT
dpi=200
path_save_png = './png_models_at_PAPA/'

# =================================
# Forcing, OSSE and observations
# =================================

# PAPA station is located at 50.1°N, 144.9°W
point_loc = [-144.9, 50.1]

# Forcing : PAPA data is hourly 
dt_forcing          = onehour      # forcing timestep

path_data = ''
name_data = ''

# Observations
period_obs          = oneday # 86400      # s, how many second between observations  

# ============================================================
# END PARAMETERS
# ============================================================