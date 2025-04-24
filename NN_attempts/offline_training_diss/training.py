import xarray as xr
import equinox as eqx
import numpy as np
import optax
import jax.numpy as jnp
import jax
import jax.tree_util as jtu

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
    
    train_set, test_set = {}, {}
    for set, data in zip([train_set, test_set],[ds_train,ds_test]):
        dUdt = data.U.astype('float64').differentiate('time')
        dVdt = data.V.astype('float64').differentiate('time')
        
        d_U,d_V = my_dynamic_RHS(data.U.values.astype('float64'), 
                                 data.V.values.astype('float64'), 
                                 data.TAx.values.astype('float64'), 
                                 data.TAy.values.astype('float64'))

        set['target'] = np.stack([dUdt - d_U, dVdt - d_V], axis=1)
        
        set['features'] = np.stack([data[key].values.astype('float64') for key in features_names], axis=1) 
        # features are first in equinox NN layers, after batch dim
        #   so features.shape = batch_dim, n_features, ydim, xdim
    
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
    first_key = list(data_set.keys())[0]
    dataset_size = data_set[first_key].shape[0] # <- time extent
    assert all(data_set[key].shape[0] == dataset_size for key in data_set.keys()) # <- check that all array are on the same time lenght
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield {key:array[batch_perm] for (key,array) in data_set.items() }
            start = end
            end = start + batch_size
 
 
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
    dim_prediction = prediction*diss_model.RENORM[1] + diss_model.RENORM[0]
    return jnp.nanmean( (dim_prediction[0] - target[0])**2 + (dim_prediction[1] - target[1])**2 )
    
def vmap_loss(dynamic_model, static_model, data_batch):
    """
    vmap of loss on the first dimension which should be the batch dimension
    """
    vmapped_loss = jax.vmap(lambda x,y:loss(dynamic_model, static_model,x,y))(data_batch['features'], data_batch['target'])
    return jnp.nanmean(vmapped_loss)


def train(
        diss_model          : eqx.Module,
        optim               : optax.GradientTransformation,
        iter_train_data     : dict,
        test_data           : dict,
        maxstep             : int,
        print_every         : int,
        save_best_model     : bool = True,
            ):
    
    dynamical_model, _ = my_partition(diss_model)
    opt_state = optim.init(dynamical_model) # eqx.filter(diss_model, eqx.is_array)

    @eqx.filter_jit
    def make_step( model, train_batch, opt_state):
        
        dynamic_model, static_model = my_partition(model)
        
        # loss_value, grads = eqx.filter_value_and_grad(vmap_loss)(model, train_batch)
        loss_value, grads = jax.value_and_grad(vmap_loss)(dynamic_model, static_model, train_batch)

        updates, opt_state = optim.update( grads, opt_state, model)

        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    Test_loss = []
    Train_loss = []
    minLoss = 999
    put_on_device(test_data) # the test dataset is small, we put it once on the GPU
    
    for step, batch_data in zip(range(maxstep), iter_train_data):

        # normalize inputs at batch size
        batch_data = normalize_features_batch(batch_data)
        put_on_device(batch_data)
        
        # get mean, std from target for renormalization of NN output
        mean, std = batch_data['target'].mean(), batch_data['target'].std()
        diss_model = eqx.tree_at( lambda t:t.RENORM, diss_model, (mean, std))
        
        model, opt_state, train_loss = make_step(diss_model, batch_data, opt_state)
        if (step % print_every) == 0 or (step == maxstep - 1):
            n_test_data = normalize_features_batch(test_data) # normalize test features
            dynamic_model, static_model = my_partition(model)
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
                bestmodel = model
    return model, bestmodel, Train_loss, Test_loss


def normalize_features_batch(batch_data):
    Nfeatures = batch_data['features'].shape[1]
    for k in range(Nfeatures):
        mean = batch_data['features'][:,k,:,:].mean()
        std = batch_data['features'][:,k,:,:].std()
        batch_data['features'][:,k,:,:] = (batch_data['features'][:,k,:,:]-mean)/std
    return batch_data

def put_on_device(batch_data):
    return jax.device_put({key:jnp.asarray(array) for key,array in batch_data.items()})

def my_partition(model):
    filter_spec = my_filter(model)
    return eqx.partition(eqx.filter(model, eqx.is_array), filter_spec) 
    
def my_filter(model):
    filter_spec = jtu.tree_map(lambda tree: False, model) # keep nothing
    filter_spec = eqx.tree_at( lambda tree: tree.layers, filter_spec, replace=True)
    return filter_spec
    