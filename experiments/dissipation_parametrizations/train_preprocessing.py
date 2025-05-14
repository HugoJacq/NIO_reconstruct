import xarray as xr
import numpy as np
import jax.numpy as jnp
import jax
from jax import lax
from copy import deepcopy
from functools import partial

def data_maker(ds               : xr.core.dataset.Dataset,
                test_ratio      : float = 20., 
                features_names  : list  = [],
                forcing_names   : list  = [],
                dx              : float = 1.,
                dy              : float = 1.,
                mydtype         : str = 'float32',):
    """ A simple data maker from a xarray dataset
    
    INPUTS:
        ds              : xarray dataset with true state
        test_ratio      : float, how much of the data is used for testing (%)
        features_names  : list of some variables availables in the files to use as features input for the NN
        forcing_names   : list of some variables availables in the files to use as forcing input for the physic-based parametrizations
                            U,V,TAx,TAy are included, forcing_names are the names of variable to add to these four.
        dx          : grid size X, in m
        dy          : grid size Y, in m
    
    OUTPUS:
        (train_set, test_set) : a tuple with data for training and validation containing target and features
    """    
    
    nt = len(ds.time)
    Ntests = test_ratio*len(ds.time)//100 # how many instants used for test
    Ntrain = nt - Ntests
    
    if Ntests>nt:
        raise Exception(f'Data_loader: you ask for {Ntests} instants for test, but the dataset contains only {nt} instants')
    
    ds_train = ds.isel(time=slice(0,Ntrain))
    ds_test = ds.isel(time=slice(-Ntests, nt))    
    train_set, test_set = {}, {}
    
    for set, data in zip([train_set, test_set],[ds_train,ds_test]):
        
        # target is true current at next time step
        set['target'] = np.stack([data.U.values.astype(mydtype),
                                  data.V.values.astype(mydtype)], axis=1)
        
        # U,V,TAx,TAy are in by default
        L_forcing = ['U','V','TAx','TAy'] + forcing_names 
        set['forcing'] = features_maker(data, L_forcing, dx, dy, out_axis=1, out_dtype=mydtype)    
        
        # features are first in equinox NN layers, after batch dim
        #   so features.shape = batch_dim, n_features, ydim, xdim
        # U and V are in by default
        L_features = ['U','V'] + features_names
        set['features'] = features_maker(data, L_features, dx, dy, out_axis=1, out_dtype=mydtype)    
                  
    return train_set, test_set




def features_maker(data             : xr.core.dataset.Dataset,
                   features_names   : list, 
                   dx               : float = 1.,
                   dy               : float = 1.,
                   out_axis         : int = 0,
                   out_dtype        : str = 'float'):
    """ A simple function that checks for names of variables in 'features_names' and then stack them
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
def normalize_batch(batch_data, L_to_be_normalized=[], deep_copy=False):
    """ A normalizer of 'batch_data'
    
    normalized = (original - mean) / std
    
    Note: only features are normalized, target isnt and so the loss is computed on terms that have a dimension
    """
    if deep_copy:
        new_data = deepcopy(batch_data)
    else:
        new_data = batch_data
    norms = {}
    # if a NN is present, normalize its inputs 
    # if NN, add 'features' to 'L_to_be_normalized'
    
    # L_names = list(batch_data.keys())
    # def fn_for_scan(carry, k):
    #     new_data, norms = carry
    #     name = L_names[k]
    #     mean, std = lax.select( name in L_to_be_normalized,
    #                            (jnp.mean(batch_data[name],axis=(0,2,3)),jnp.std(batch_data[name],axis=(0,2,3))),
    #                            (jnp.zeros(batch_data[name].shape[1]),jnp.ones(batch_data[name].shape[1])))
    #     new_data[name] = (batch_data[name]-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis] 
    #     norms[name] = {'mean':mean, 'std':std}
    #     return new_data, norms
    
    # new_data, norms = lax.scan(fn_for_scan, (new_data, norms), jnp.arange(len(L_names)))
    
    
    for name in list(batch_data.keys()):
        if name in L_to_be_normalized:
            mean = jnp.mean(batch_data[name],axis=(0,2,3))
            std = jnp.std(batch_data[name],axis=(0,2,3))
            new_data[name] = (batch_data[name]-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis] 
            norms[name] = {'mean':mean, 'std':std}
        else:
            norms[name] = {'mean':jnp.zeros(batch_data[name].shape[1]), 
                           'std':jnp.ones(batch_data[name].shape[1])} # do nothing (case when no NN)
    
    return new_data, norms
    
    
    
def batch_loader(data_set   : dict, 
                 batch_size : int):
    """ A simpe batch loader that gives a time batch of size 'batch_size', after a shuffle of indices
    
    INPUTS:
        data_set    : dict
        batch_size  : int
    OUPUT:
        a iterator that gives a batch of size 'batch_size' each time its called.
    """
    first_key = list(data_set.keys())[0]
    dataset_size = data_set[first_key].shape[0] # <- time extent
    assert all(data_set[key].shape[0] == dataset_size for key in data_set.keys()) # <- check that all array are on the same time lenght
    indices = np.arange(dataset_size)
    if batch_size>len(indices):
        raise Exception(f'Your batch size is {batch_size} but you have data for only {len(indices)} points')
    while True:
        perm = np.random.permutation(indices)
        start = 0
        if batch_size<=0:
            end = dataset_size # no batch
            mybatch = dataset_size
        else:
            end = batch_size
            mybatch = batch_size
        while start < dataset_size:
            print(start, end)
            batch_perm = perm[start:end]            
            yield {key:array[batch_perm] for (key,array) in data_set.items() }
            start = end
            end = start + mybatch
            if end > dataset_size:
                end = dataset_size