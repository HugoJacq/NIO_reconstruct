import xarray as xr
import equinox as eqx
import numpy as np
import optax
import jax.numpy as jnp
import jax
import jax.tree_util as jtu
from copy import deepcopy

from NNmodels import Rayleigh_damping

def data_maker(ds              : xr.core.dataset.Dataset,
                Ntests          : int, 
                features_names  : list,
                my_dynamic_RHS  : eqx.Module):
    """
    INPUTS:
        filenames   : list of strings, names of files to use as inputs
        Ntests      : integer, how many instants for test data
        features_names  : list of some variables availables in the files to use as features input for the NN
        Nsize       : int, number of xy points to use
    
    OUTPUS:
        (train_set, test_set) : a tuple with data for training and validation, target and features
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
        dUdt = data.U.differentiate('time').astype(mydtype)
        dVdt = data.V.differentiate('time').astype(mydtype)
        
        d_U,d_V = my_dynamic_RHS(data.U.values.astype(mydtype), 
                                 data.V.values.astype(mydtype),  
                                 data.TAx.values.astype(mydtype),  
                                 data.TAy.values.astype(mydtype)) 

        set['target'] = np.stack([dUdt - d_U, dVdt - d_V], axis=1)
        
        # features are first in equinox NN layers, after batch dim
        #   so features.shape = batch_dim, n_features, ydim, xdim
        set['features'] = np.stack([data[key].values.astype(mydtype) for key in features_names], axis=1)  # .astype('float64')
        
    
    return train_set, test_set
         
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
    print(data_set['target'].shape)

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
 
 
def loss(dynamic_model, static_model, n_features, target):
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
    print(prediction.shape)
    #dim_prediction = prediction*diss_model.RENORMstd[:,np.newaxis, np.newaxis] + diss_model.RENORMmean[:,np.newaxis, np.newaxis]
    # return jnp.sqrt( (dim_prediction[0] - target[0])**2 + (dim_prediction[1] - target[1])**2 )
    return jnp.sqrt( (prediction[0] - target[0])**2 + (prediction[1] - target[1])**2 )

def vmap_loss(dynamic_model, static_model, data_batch):
    """
    vmap of loss on the first dimension which should be the batch dimension
    """
    vmapped_loss = jax.vmap(loss, in_axes=(None, None, 0, 0))(dynamic_model, static_model, data_batch['features'], data_batch['target'])
    print(vmapped_loss.shape)
    return jnp.nanmean(vmapped_loss)


def train(
        diss_model          : eqx.Module,
        optim               : optax.GradientTransformation,
        iter_train_data     : dict,
        train_data          : dict,
        test_data           : dict,
        maxstep             : int,
        print_every         : int,
        save_best_model     : bool = True,
            ):
    
    

    @eqx.filter_jit
    def make_step( model, train_batch, opt_state):
        
        # dynamic_model, static_model = my_partition(model)
        
        loss_value, grads = jax.value_and_grad(vmap_loss)(model, static_model, train_batch)
        # print(model)
        # loss_value = vmap_loss(model, static_model, train_batch)
        # print(model)
        # grads = jax.jacfwd(vmap_loss)(model, static_model, train_batch)

        # jax.debug.print('BEFORE opt_state mu R {}', opt_state[0].mu.layers[0].R)
        # jax.debug.print('BEFORE opt_state nu R {}', opt_state[0].nu.layers[0].R)
        print(model.layers[0].R)
        print(grads)
        updates, opt_state = optim.update(grads, opt_state, model) # 
        # jax.debug.print('AFTER opt_state mu R {}', opt_state[0].mu.layers[0].R)
        # jax.debug.print('AFTER opt_state nu R {}', opt_state[0].nu.layers[0].R)
        
        # jax.debug.print('update to apply on R {}', updates.layers[0].R)
        # jax.debug.print('grad of bias layer 1 {}', grads.layers[0].bias)
        # jax.debug.print('grad of R {}', grads.layers[0].R)
        
        new_dyn = eqx.apply_updates(model, updates)
        combined = eqx.combine(new_dyn, static_model)
        # jax.debug.print('new R : {}, {}', combined.layers[0].R, grads.layers[0].R)
        # jax.debug.print('R,  : {}',model.layers[0].R)
        # newmodel = eqx.combine(dynamic_model, static_model)
        # newmodel = eqx.tree_at( lambda t:t.layers[0].R, newmodel, jnp.asarray(0.15, dtype='float') )
        # jax.debug.print('new R : {}', newmodel.layers[0].R)
        return combined, opt_state, loss_value
    
    
    dynamic_model, static_model = my_partition(diss_model)
    print(dynamic_model.layers[0].R)
    opt_state = optim.init(dynamic_model) # eqx.filter(diss_model, eqx.is_array)
    
    
    Test_loss = []
    Train_loss = []
    minLoss = 999
    put_on_device(test_data) # the test dataset is small, we put it once on the GPU
        
    # get mean, std from target for renormalization of NN output
    mean, std = jnp.mean(train_data['target'],axis=(0,2,3)), jnp.std(train_data['target'],axis=(0,2,3))
    diss_model = eqx.tree_at( lambda t:t.RENORMmean, diss_model, mean)
    diss_model = eqx.tree_at( lambda t:t.RENORMstd, diss_model, std)
    
    
    #for step in range(maxstep):
    for step, batch_data in zip(range(maxstep), iter_train_data):    
    
        # normalize inputs at batch size
        batch_data = normalize_batch(batch_data)
        put_on_device(batch_data)
        
        dynamic_model, opt_state, train_loss = make_step(dynamic_model, batch_data, opt_state)
        # dynamic_model, static_model = my_partition(model)
        
        if (step % print_every) == 0 or (step == maxstep - 1):
            n_test_data = normalize_batch(test_data) # normalize test features
            test_loss = vmap_loss(dynamic_model, static_model, n_test_data)
            print(
                f"{step=}, train_loss={train_loss.item()}, "    # train loss of current epoch (uses the old model)
                f"test_loss={test_loss.item()}"                 # test loss of next epoch (uses the new model)
            )
        Test_loss.append(test_loss)
        Train_loss.append(train_loss)  
        if save_best_model:
            if test_loss<minLoss:
                # keep the best model
                minLoss = test_loss
                bestmodel = eqx.combine(dynamic_model, static_model)
        lastmodel = eqx.combine(dynamic_model, static_model)
    return lastmodel, bestmodel, Train_loss, Test_loss


def normalize_batch(batch_data):
    new_data = deepcopy(batch_data)
    # features
    Nfeatures = batch_data['features'].shape[1]
    for k in range(Nfeatures):
        mean = np.mean(batch_data['features'][:,k,:,:])
        std = np.std(batch_data['features'][:,k,:,:])
        new_data['features'][:,k,:,:] = (batch_data['features'][:,k,:,:]-mean)/std
    # target
    Ntarget = batch_data['target'].shape[1]
    for k in range(Ntarget):
        mean = np.mean(batch_data['target'][:,k,:,:])
        std = np.std(batch_data['target'][:,k,:,:])
        new_data['target'][:,k,:,:] = (batch_data['target'][:,k,:,:]-mean)/std        
    return new_data

def put_on_device(batch_data):
    return jax.device_put({key:jnp.asarray(array) for key,array in batch_data.items()})

def my_partition(model):
    filter_spec = my_filter(model)
    return eqx.partition(model, filter_spec) 
    
def my_filter(model):
    filter_spec = jtu.tree_map(lambda x: False, model)
    
    # Mark all weights and biases in Linear layers as trainable
    for i, layer in enumerate(model.layers):
        if isinstance(layer, eqx.nn.Linear) or isinstance(layer, eqx.nn.Conv):
            filter_spec = eqx.tree_at(lambda t: t.layers[i].weight, filter_spec, replace=True)
            filter_spec = eqx.tree_at(lambda t: t.layers[i].bias, filter_spec, replace=True)
        elif isinstance(layer, Rayleigh_damping):
            filter_spec = eqx.tree_at(lambda t: t.layers[0].R, filter_spec, replace=True)
    return filter_spec
    