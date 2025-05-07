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
from jaxtyping import PyTree


from NNmodels import Rayleigh_damping
from preprocessing import normalize_batch
  
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
        perm = np.random.permutation(indices) # indices # 
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
 

# ================================================================================= 
# LOSS DEFINITION

# -> MODEL WITH NN ONLY
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


@eqx.filter_jit
def evaluate_NN(dynamic_model, static_model, data_batch):
    return vmap_loss(dynamic_model, static_model, data_batch)

# -> HYBRID MODELS
def loss_hybrid(dynamic_model, static_model, n_features, n_target, n_forcing):
    """
    this loss is computed on normalized values
    """
    RHS_model = eqx.combine(dynamic_model, static_model)
    
    
    prediction = RHS_model(n_features, n_forcing)
    
    # prediction_diss_undim = RHS_model.dissipationNN(n_features)
    
    # giving back a dimension to NN term
    # prediction_diss_dim = prediction_diss_undim*RHS_model.RENORMstd[:,np.newaxis, np.newaxis] + RHS_model.RENORMmean[:,np.newaxis, np.newaxis] 
    
    # stress term with normalized forcing
    # prediction_stress = RHS_model.stress(n_forcing[0],n_forcing[1])
    
    # regularization
    alpha = 0.01
    # prediction = alpha*prediction_stress + (1-alpha)*prediction_diss_undim
    
    # prediction = prediction_stress + prediction_diss_undim
    # jax.debug.print('normalized: stress {}, diss {}, target {}', alpha*prediction_stress[0].mean(), (1-alpha)*prediction_diss_undim[0].mean(), n_target[0].mean())
    return jnp.sqrt( (prediction[0] - n_target[0])**2 + (prediction[1] - n_target[1])**2 )

def vmap_loss_hybrid(dynamic_model, static_model, data_batch):
    """
    """ 
    vmapped_loss = jax.vmap(
                            loss_hybrid, in_axes=(None, None, 0, 0, 0)
                                )(dynamic_model, static_model, 
                                    data_batch['features'], 
                                    data_batch['target'],
                                    data_batch['forcing'])
    return jnp.nanmean(vmapped_loss)   

@eqx.filter_jit
def evaluate_hybrid(dynamic_model, static_model, data_batch):
    return vmap_loss_hybrid(dynamic_model, static_model, data_batch)
# ================================================================================= 





def train(
        the_model          : eqx.Module,
        optim               : optax.GradientTransformation,
        iter_train_data     : dict,
        train_data          : dict,
        test_data           : dict,
        maxstep             : int,
        print_every         : int,
        restarting          : bool= False,
        mode                : str = 'NN_only',
        save_best_model     : bool = True,
        saved_opt_state     : PyTree = None,
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
        # jax.debug.print('K at entry make_step_hybrid {}',model.stress.K)
        loss_value, grads = jax.value_and_grad(vmap_loss_hybrid)(model, static_model, train_batch)
        # loss_value = vmap_loss(model, static_model, train_batch)
        # grads = jax.jacfwd(vmap_loss)(model, static_model, train_batch)
        updates, opt_state = optim.update(grads, opt_state, model) #      
        # jax.debug.print('K at middle make_step_hybrid {}',updates.stress.K) 
        new_dyn = eqx.apply_updates(model, updates)
        # jax.debug.print('K at end make_step_hybrid {}',new_dyn.stress.K)
        return new_dyn, opt_state, loss_value, grads
    
    # =================
    # INITIALIZITION
    # =================
    dynamic_model, static_model = my_partition(the_model, mode=mode)
    if restarting:
        opt_state = saved_opt_state
    else:
        opt_state = optim.init(dynamic_model)
    
    Test_loss = []
    Train_loss = []
    minLoss = 999
    put_on_device(test_data) # the test dataset is small, we put it once on the GPU
        
        
    bestdyn = dynamic_model
    opt_state_save = opt_state
        
    # =================
    # TRAIN LOOP
    # =================
    for step, batch_data in zip(range(maxstep), iter_train_data):    
       
        ###################
        # DE-NORMALIZATION
        ###################
        # get mean, std from target for renormalization of NN output
        mean, std = jnp.mean(batch_data['target'],axis=(0,2,3)), jnp.std(batch_data['target'],axis=(0,2,3))
        static_model = eqx.tree_at( lambda t:t.RENORMmean, static_model, mean)
        static_model = eqx.tree_at( lambda t:t.RENORMstd, static_model, std)
       
        #########################
        # NORMALIZATION AT BATCH
        #########################
        batch_data = normalize_batch(batch_data) # <- modify in place 'batch_data'
        ###################
        # MAKE STEP
        ###################
        if mode=='NN_only':
            dynamic_model, opt_state, train_loss, _ = make_step(dynamic_model, batch_data, opt_state)
            evaluate = evaluate_NN
        
        elif mode=='hybrid':
            dynamic_model, opt_state, train_loss, grads = make_step_hybrid(dynamic_model, batch_data, opt_state)
            if grads.stress.K is not None:
                # dK = jnp.abs(grads.stress.K*jnp.exp(dynamic_model.stress.K))
                dK = jnp.abs(grads.stress.K)
                if dK < 1e-5:
                    print('     K0 has converged, it is now kept constant')
                    print(f'         K0 = {dynamic_model.stress.K}, dK0 = {grads.stress.K}')
                    static_model = eqx.tree_at( lambda t:t.stress.K, static_model, dynamic_model.stress.K, is_leaf=lambda x: x is None) # <- set K0 as converged
                    dynamic_model = eqx.tree_at( lambda t:t.stress.K, dynamic_model, None) # <- remove K0 from trainable parameters
                else:
                    print(f'         K0 = {dynamic_model.stress.K}, dK0 = {grads.stress.K}')
            evaluate = evaluate_hybrid
            
        elif mode=='hybrid_sequential':
            print('sequential model to be done')
        else:
            raise Exception(f'You chose the training mode {mode} but it is not recognized')

        
        ###################
        # LOSS PLOT DATA
        ###################
        if (step % print_every) == 0 or (step == maxstep - 1):
            n_test_data = normalize_batch(test_data) # normalize test features
            test_loss = evaluate(dynamic_model, static_model, n_test_data)
            print(
                f"{step=}, train_loss={train_loss.item()}, "    # train loss of current epoch (uses the old model)
                f"test_loss={test_loss.item()}"                 # test loss of next epoch (uses the new model)
            )
            Test_loss.append(test_loss.item())
        Train_loss.append(train_loss.item())  
        
        ###################
        # SAVING BEST MODEL
        ###################
        
        if save_best_model:
            if test_loss<minLoss:
                # keep the best model
                minLoss = test_loss
                bestdyn = dynamic_model 
                opt_state_save = opt_state
        if step==maxstep-1:
            lastmodel = eqx.combine(dynamic_model, static_model)
            bestmodel = eqx.combine(bestdyn, static_model)        

    return lastmodel, bestmodel, Train_loss, Test_loss, opt_state_save





def put_on_device(batch_data):
    return jax.device_put({key:jnp.asarray(array) for key,array in batch_data.items()})

def my_partition(model, mode='NN_only'):
    filter_spec = jtu.tree_map(lambda x: False, model)
    if mode=='NN_only':
        filter_spec = my_filter_NN(model, filter_spec)
    elif mode=='hybrid':
        NN_filter = my_filter_NN(model.dissipationNN, filter_spec.dissipationNN)
        filter_spec = eqx.tree_at(lambda t: t.dissipationNN, filter_spec, NN_filter)
        filter_spec = eqx.tree_at(lambda t: t.stress.K, filter_spec, replace=True)
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
    