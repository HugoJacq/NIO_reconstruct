import xarray as xr
import equinox as eqx
import numpy as np
import optax
import jax.numpy as jnp
import jax
import jax.tree_util as jtu
from copy import deepcopy
import time as clock
from functools import partial


def data_maker(ds              : xr.core.dataset.Dataset,
                Ntests          : int, 
                features_names  : list,
                no_parameter_RHS  : eqx.Module,
                dx              : float,
                dy              : float,
                dt_forcing      : float,
                mode            : str = 'NN_only'):
    """
    INPUTS:
        filenames   : list of strings, names of files to use as inputs
        Ntests      : integer, how many instants for test data
        features_names  : list of some variables availables in the files to use as features input for the NN
        Nsize       : int, number of xy points to use
        dx          : grid size X, in m
        dy          : grid size Y, in m
    
    OUTPUS:
        (train_set, test_set) : a tuple with data for training and validation containing target and features
    """    
    nt = len(ds.time)

    Ntrain = nt-Ntests
    
    if Ntests>nt:
        raise Exception(f'Data_loader: you ask for {Ntests} instants for test, but the dataset contains only {nt} instants')
    
    ds_train = ds.isel(time=slice(0,Ntrain))
    ds_test = ds.isel(time=slice(-Ntests, -1))    
    mydtype = 'float'
    train_set, test_set = {}, {}
    for set, data in zip([train_set, test_set],[ds_train,ds_test]):
        dUdt = np.gradient(data.U, dt_forcing,axis=0).astype(mydtype)
        dVdt = np.gradient(data.V, dt_forcing,axis=0).astype(mydtype)
        
        
        if mode=='NN_only':
            # here 'no_parameter_RHS' is the Coriolis term and the Stress term
            d_U,d_V = no_parameter_RHS(data.U.values.astype(mydtype), 
                                     data.V.values.astype(mydtype),  
                                     data.TAx.values.astype(mydtype),  
                                     data.TAy.values.astype(mydtype)) 

            set['target'] = np.stack([dUdt - d_U, dVdt - d_V], axis=1)
        
        elif mode=='hybrid':
            # here 'no_parameter_RHS' is the Coriolis term
            d_U, d_V = no_parameter_RHS(data.U.values.astype(mydtype),  
                                        data.V.values.astype(mydtype))
            set['target'] = np.stack([dUdt-d_U, dVdt-d_V], axis=1)
            set['forcing'] = features_maker(data, ['TAx','TAy'], out_axis=1, out_dtype=mydtype) # forcing is for the RHS_dynamic part
        
        # features are first in equinox NN layers, after batch dim
        #   so features.shape = batch_dim, n_features, ydim, xdim
        set['features'] = features_maker(data, features_names, dx, dy, out_axis=1, out_dtype=mydtype)
        
                                
    return train_set, test_set

def features_maker(data         : xr.core.dataset.Dataset,
                   features_names:list, 
                   dx           :float = 1.,
                   dy           :float = 1.,
                   out_axis     :int = 0,
                   out_dtype    :str = 'float'):
    """
    A simple function that checks for names of variables in 'features_names' and then stack them
        for use of input for a neural network.
        
        The name can be a gradient if starting with 'grad' then a direction (e.g. 'graxU')
    """
    av_vars = list(data.data_vars)
    L_f = []
    for _,feature in enumerate(features_names):
        if feature in av_vars:
            L_f.append( data[feature].values.astype(out_dtype) )
        elif feature[:4]=='grad' and feature[5:] in av_vars:
            if feature[4]=='x':
                DX, axis = dx, -1
            elif feature[4]=='y':
                DX, axis = dy, -2           
            L_f.append( np.gradient(data[feature[5:]], DX, axis=axis).astype(out_dtype) )
        else:
            raise Exception(f'You want to use the variables {feature} as a feature but it is not recognized')
    return np.stack(L_f, axis=out_axis)
   
@partial(jax.jit, static_argnames=['deep_copy'])
def normalize_batch(batch_data, deep_copy=False):
    """
    normalized = (original - mean) / std
    """
    if deep_copy:
        new_data = deepcopy(batch_data)
    else:
        new_data = batch_data
        
    # print(batch_data['target'].shape)
    L_to_be_normalized = ['features']  # list(batch_data.keys())    
    for name in L_to_be_normalized:
        mean = jnp.mean(batch_data[name],axis=(0,2,3))
        std = jnp.std(batch_data[name],axis=(0,2,3))
        new_data[name] = (batch_data[name]-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis] 
        
        # Num = batch_data[name].shape[1]
        # for k in range(Num):
        #     mean = jnp.mean(batch_data[name][:,k,:,:])
        #     std = jnp.std(batch_data[name][:,k,:,:])
        #     new_data[name] = new_data[name].at[:,k,:,:].set( (batch_data[name][:,k,:,:]-mean)/std )   
    return new_data