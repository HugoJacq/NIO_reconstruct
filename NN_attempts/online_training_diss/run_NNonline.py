"""
Goal: 
    Estimate ageostrophic current, from a simple slab model. Given wind stress, geostrophy (sst ?)


Context:

    the slab equation in complex notation is:

        dC/dt = -i.fc.C + K0.Tau - dissipation              (1)

    with C = U+i.V
         Tau = Taux + i.Tauy
         K0 = real
         fc = Coriolis frequency
         
    the dissipation term is usually parametrized as a Rayleigh damping term rC with r a constant.
    
Our approach:

    we think that using a Rayleigh damping term is very restrictive. Seeking for more a more expressive term, 
    we try to model it using a neural network.
    
    either the full dissipation as a NN(Ug, U, Tau), or just the r as a constant, or the r(Ug, U, Tau) as a function of features 

This script:

    We aim to have a dissipation term that allow for a prediction of the RHS of (1), then integrate this one or multiple time to get 
        a surface current trajectory.
        
    This script is online training as we compare the trajectory output from a hybrid model with the true trajectory.
    The NN (= dissipation term) is used as a part of the RHS of (1) that we integrate in time.
        
        Loss = || true trajectory - estimated trajectory ||
        
            with estimated trajectory = model applied k times on initial condition
"""
# regular modules import
import xarray as xr
import numpy as np
import jax
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import os 
import sys
sys.path.insert(0, '../../src')
# jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # for jax
# jax.config.update("jax_enable_x64", True)

# my modules imports
from training import data_maker, batch_loader, train, normalize_batch, features_maker
from models_with_NN import *
from constants import oneday, distance_1deg_equator