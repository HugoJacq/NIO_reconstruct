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

from NNmodels import Rayleigh_damping
from preprocessing import normalize_batch

# def data_maker(ds              : xr.core.dataset.Dataset,
#                 Ntests          : int, 
#                 features_names  : list,
#                 my_dynamic_RHS  : eqx.Module,
#                 dx              : float,
#                 dy              : float,
#                 mode            : str = 'NN_only'):
#     """
#     INPUTS:
#         filenames   : list of strings, names of files to use as inputs
#         Ntests      : integer, how many instants for test data
#         features_names  : list of some variables availables in the files to use as features input for the NN
#         Nsize       : int, number of xy points to use
#         dx          : grid size X, in m
#         dy          : grid size Y, in m
    
#     OUTPUS:
#         (train_set, test_set) : a tuple with data for training and validation containing target and features
#     """    
#     nt = len(ds.time)

#     Ntrain = nt-Ntests
    
#     if Ntests>nt:
#         raise Exception(f'Data_loader: you ask for {Ntests} instants for test, but the dataset contains only {nt} instants')
    
#     ds_train = ds.isel(time=slice(0,Ntrain))
#     ds_test = ds.isel(time=slice(-Ntests, -1))    
#     mydtype = 'float'
#     train_set, test_set = {}, {}
#     for set, data in zip([train_set, test_set],[ds_train,ds_test]):
#         dUdt = np.gradient(data.U, 3600.,axis=0).astype(mydtype)
#         dVdt = np.gradient(data.V, 3600.,axis=0).astype(mydtype)
        
        
#         if mode=='NN_only':
#             d_U,d_V = my_dynamic_RHS(data.U.values.astype(mydtype), 
#                                      data.V.values.astype(mydtype),  
#                                      data.TAx.values.astype(mydtype),  
#                                      data.TAy.values.astype(mydtype)) 

#             set['target'] = np.stack([dUdt - d_U, dVdt - d_V], axis=1)
        
#         elif mode=='hybrid':
#             set['target'] = np.stack([dUdt, dVdt], axis=1)
#             set['forcing'] = features_maker(data, ['U','V','TAx','TAy'], out_axis=1,out_dtype=mydtype) # forcing is for the RHS_dynamic part
        
#         # features are first in equinox NN layers, after batch dim
#         #   so features.shape = batch_dim, n_features, ydim, xdim
#         set['features'] = features_maker(data, features_names,dx,dy,out_axis=1,out_dtype=mydtype)
        
                                
#     return train_set, test_set

# def features_maker(data         : xr.core.dataset.Dataset,
#                    features_names:list, 
#                    dx           :float = 1.,
#                    dy           :float = 1.,
#                    out_axis     :int = 0,
#                    out_dtype    :str = 'float'):
#     """
#     A simple function that checks for names of variables in 'features_names' and then stack them
#         for use of input for a neural network.
        
#         The name can be a gradient if starting with 'grad' then a direction (e.g. 'graxU')
#     """
#     av_vars = list(data.data_vars)
#     L_f = []
#     for _,feature in enumerate(features_names):
#         if feature in av_vars:
#             L_f.append( data[feature].values.astype(out_dtype) )
#         elif feature[:4]=='grad' and feature[5:] in av_vars:
#             if feature[4]=='x':
#                 DX, axis = dx, -1
#             elif feature[4]=='y':
#                 DX, axis = dy, -2           
#             L_f.append( np.gradient(data[feature[5:]], DX, axis=axis).astype(out_dtype) )
#         else:
#             raise Exception(f'You want to use the variables {feature} as a feature but it is not recognized')
#     return np.stack(L_f, axis=out_axis)
   
   
def batch_loader(data_set   : dict, 
                 batch_size : int):
    """
    A simpe batch loader that gives a time batch of size 'batch_size', after a shuffle of indices
    
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
    while True:
        perm = indices # np.random.permutation(indices)
        start = 0
        if batch_size<=0:
            end = dataset_size-1 # no batch
            mybatch = dataset_size-1
        else:
            end = batch_size
            mybatch = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]            
            yield {key:array[batch_perm] for (key,array) in data_set.items() }
            start = end
            end = start + mybatch
            if end > dataset_size:
                end = dataset_size
 
 
def loss(dynamic_model, static_model, n_features, n_target):
    """
    INPUTS:
        diss_model  : NN, eats normalized inputs, outputs non dimensionnal dissipation
                        the model contains also mean,std to give a dimension to the NN output
        n_features  : an array of shape (Nfeatures, ny, nx)
        target      : target of shape (2, ny, nx)
    OUTPUTS:
        scalar value of the loss
    """
    diss_model = eqx.combine(dynamic_model, static_model)
    prediction = diss_model(n_features)
    return jnp.sqrt( (prediction[0] - n_target[0])**2 + (prediction[1] - n_target[1])**2 )


def vmap_loss(dynamic_model, static_model, data_batch):
    """
    vmap of loss on the first dimension which should be the batch dimension
    """
    vmapped_loss = jax.vmap(
                            loss, in_axes=(None, None, 0, 0)
                                )(dynamic_model, 
                                  static_model, 
                                  data_batch['features'], 
                                  data_batch['target'])
    return jnp.nanmean(vmapped_loss)

# @partial(eqx.filter_jit, static_argnames=['static_model','data_batch'])
@eqx.filter_jit
def evaluate(dynamic_model, static_model, data_batch):
    return vmap_loss(dynamic_model, static_model, data_batch)

def loss_hybrid(dynamic_model, static_model, n_features, n_target, n_forcing):
    """
    """
    RHS_model = eqx.combine(dynamic_model, static_model)
    prediction = RHS_model(n_features, n_forcing)
    return jnp.sqrt( (prediction[0] - n_target[0])**2 + (prediction[1] - n_target[1])**2 )

def vmap_loss_hybrid(dynamic_model, static_model, data_batch):
    """
    """ 
    vmapped_loss = jax.vmap(
                            loss_hybrid, in_axes=(None, None, 0, 0)
                                )(dynamic_model, static_model, 
                                    data_batch['features'], 
                                    data_batch['target'],
                                    data_batch['forcing'])
    return jnp.nanmean(vmapped_loss)   
    

def train(
        the_model          : eqx.Module,
        optim               : optax.GradientTransformation,
        iter_train_data     : dict,
        train_data          : dict,
        test_data           : dict,
        maxstep             : int,
        print_every         : int,
        mode                : str = 'NN_only',
        save_best_model     : bool = True,
            ):
    
    @eqx.filter_jit
    def make_step( model, train_batch, opt_state): #  <- train theta only
        loss_value, grads = jax.value_and_grad(vmap_loss)(model, static_model, train_batch)
        # loss_value = vmap_loss(model, static_model, train_batch)
        # grads = jax.jacfwd(vmap_loss)(model, static_model, train_batch)
        updates, opt_state = optim.update(grads, opt_state, model) #         
        new_dyn = eqx.apply_updates(model, updates)
        return new_dyn, opt_state, loss_value, grads
    
    @eqx.filter_jit
    def make_step_hybrid(model, train_batch, opt_state): #  <- train both K0 and theta
        loss_value, grads = jax.value_and_grad(vmap_loss_hybrid)(model, static_model, train_batch)
        # loss_value = vmap_loss(model, static_model, train_batch)
        # grads = jax.jacfwd(vmap_loss)(model, static_model, train_batch)
        updates, opt_state = optim.update(grads, opt_state, model) #         
        new_dyn = eqx.apply_updates(model, updates)
        return new_dyn, opt_state, loss_value, grads
    
    
    dynamic_model, static_model = my_partition(the_model, mode=mode)

    opt_state = optim.init(dynamic_model)
    
    
    Test_loss = []
    Train_loss = []
    minLoss = 999
    put_on_device(test_data) # the test dataset is small, we put it once on the GPU

    # get mean, std from target for renormalization of NN output
    mean, std = jnp.mean(train_data['target'],axis=(0,2,3)), jnp.std(train_data['target'],axis=(0,2,3))
    static_model = eqx.tree_at( lambda t:t.RENORMmean, static_model, mean)
    static_model = eqx.tree_at( lambda t:t.RENORMstd, static_model, std)
    
    
    #for step in range(maxstep):
    for step, batch_data in zip(range(maxstep), iter_train_data):    
        # normalize inputs at batch size
        batch_data = normalize_batch(batch_data)
        
        if mode=='NN_only':
            dynamic_model, opt_state, train_loss, _ = make_step(dynamic_model, batch_data, opt_state)
        
        elif mode=='hybrid':
            dynamic_model, opt_state, train_loss, grads = make_step_hybrid(dynamic_model, batch_data, opt_state)
            if grads.RHS_dyn.K < 1e-5:
                static_model = eqx.tree_at( lambda t:t.grads.RHS_dyn.K, static_model, dynamic_model.RHS_dyn.K) # <- set K0 as converged
                dynamic_model = eqx.tree_at( lambda t:t.grads.RHS_dyn.K, dynamic_model, None) # <- remove K0 from trainable parameters
            
        elif mode=='hybrid_sequential':
            print('sequential model to be done')
        else:
            raise Exception(f'You chose the training mode {mode} but it is not recognized')



        if (step % print_every) == 0 or (step == maxstep - 1):
            n_test_data = normalize_batch(test_data) # normalize test features
            test_loss = evaluate(dynamic_model, static_model, n_test_data)
            print(
                f"{step=}, train_loss={train_loss.item()}, "    # train loss of current epoch (uses the old model)
                f"test_loss={test_loss.item()}"                 # test loss of next epoch (uses the new model)
            )
            Test_loss.append(test_loss.item())
        Train_loss.append(train_loss.item())  
        if save_best_model:
            if test_loss<minLoss:
                # keep the best model
                minLoss = test_loss
                bestdyn = dynamic_model 
        else:
            bestdyn = dynamic_model
            
        if step==maxstep-1:
            lastmodel = eqx.combine(dynamic_model, static_model)
            bestmodel = eqx.combine(bestdyn, static_model)

    return lastmodel, bestmodel, Train_loss, Test_loss

# @partial(jax.jit, static_argnames=['deep_copy'])
# def normalize_batch(batch_data, deep_copy=False):
#     """
#     normalized = (original - mean) / std
#     """
#     if deep_copy:
#         new_data = deepcopy(batch_data)
#     else:
#         new_data = batch_data
        
#     L_to_be_normalized = ['features','target']    
#     for name in L_to_be_normalized:
#         Num = batch_data[name].shape[1]
#         for k in range(Num):
#             mean = jnp.mean(batch_data[name][:,k,:,:])
#             std = jnp.std(batch_data[name][:,k,:,:])
#             new_data[name] = new_data[name].at[:,k,:,:].set( (batch_data[name][:,k,:,:]-mean)/std )   
#     return new_data

def put_on_device(batch_data):
    return jax.device_put({key:jnp.asarray(array) for key,array in batch_data.items()})

def my_partition(model, mode='NN_only'):
    filter_spec = jtu.tree_map(lambda x: False, model)
    if mode=='NN_only':
        filter_spec = my_filter_NN(model, filter_spec)
    elif mode=='hybrid':
        NN_filter = my_filter_NN(model.dissipationNN, filter_spec.dissipationNN)
        filter_spec = eqx.tree_at(lambda t: t.dissipationNN, filter_spec, NN_filter)
        filter_spec = eqx.tree_at(lambda t: t.RHS_dyn.K, filter_spec, replace=True)
    return eqx.partition(model, filter_spec) 
     
     
     
def my_filter_NN(model, filter_spec):   
    # Mark all weights and biases in Linear layers as trainable
    for i, layer in enumerate(model.layers):
        if isinstance(layer, eqx.nn.Linear) or isinstance(layer, eqx.nn.Conv):
            filter_spec = eqx.tree_at(lambda t: t.layers[i].weight, filter_spec, replace=True)
            filter_spec = eqx.tree_at(lambda t: t.layers[i].bias, filter_spec, replace=True)
        elif isinstance(layer, Rayleigh_damping):
            filter_spec = eqx.tree_at(lambda t: t.layers[0].R, filter_spec, replace=True)
    return filter_spec
    