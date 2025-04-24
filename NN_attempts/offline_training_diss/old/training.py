import optax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree, Float, Int, Array
import jax.tree_util as jtu




def my_partition(model):
    filter_spec = my_filter(model)
    return eqx.partition(eqx.filter(model, eqx.is_array), filter_spec) 

def my_filter(model):
    filter_spec = jtu.tree_map(lambda tree: False, model) # keep nothing
    filter_spec = eqx.tree_at( lambda tree: tree.K, filter_spec, replace=True)
    filter_spec = eqx.tree_at( lambda tree: tree.dissipation, filter_spec, replace=True)
    return filter_spec

def loss(dynamic_model, static_model, data_train):
    mymodel = eqx.combine(dynamic_model, static_model)
    pred_RHS = jax.vmap(mymodel)(data_train['U'], 
                                 data_train['V'], 
                                 data_train['TAx'], 
                                 data_train['TAy'],
                                 data_train['features'])
    # jax.debug.print('dUdt, d_U: {}, {}',data_train['dUdt'][0,5,5], pred_RHS[0][0,5,5])
    # jax.debug.print('dVdt, d_V: {}, {}',data_train['dVdt'][0,5,5], pred_RHS[1][0,5,5])
    return jnp.nanmean( jnp.sqrt((pred_RHS[0]-data_train['dUdt'])**2 + (pred_RHS[1]-data_train['dVdt'])**2 ))
    


def evaluate(model, test_data):
    pred_RHS = jax.vmap(model)(test_data['U'], 
                                 test_data['V'], 
                                 test_data['TAx'], 
                                 test_data['TAy'],
                                 test_data['features'])
    return jnp.nanmean( jnp.sqrt((pred_RHS[0]-test_data['dUdt'])**2 + (pred_RHS[1]-test_data['dVdt'])**2 ))

# train process
def train(
        model: eqx.Module,
        optim: optax.GradientTransformation,
        iter_train_data: dict,
        test_data: dict,
        maxstep: Int,
        print_every: Int
            ):
    
    dynamic_model, static_model = my_partition(model)
    opt_state = optim.init( dynamic_model )

    @eqx.filter_jit
    def make_step( model, train_batch, opt_state: PyTree):
        
        dynamic_model, static_model = my_partition(model)
        
        #jax.debug.print('before grad compute K0: {}', dynamic_model.K0)
                
        loss_value, grads = eqx.filter_value_and_grad(loss)(dynamic_model, static_model, train_batch)

        updates, opt_state = optim.update( grads, opt_state, model)

        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    Test_loss = []
    Train_loss = []
    minLoss = 999
    for step, batch_data in zip(range(maxstep), iter_train_data):
        model, opt_state, train_loss = make_step(model, batch_data, opt_state)
        if (step % print_every) == 0 or (step == maxstep - 1):
            test_loss = evaluate(model, test_data)
            print(
                f"{step=}, train_loss={train_loss.item()}, "    # train loss of current epoch (uses the old model)
                f"test_loss={test_loss.item()}"                 # test loss of next epoch (uses the new model)
            )
        Test_loss.append(test_loss)
        Train_loss.append(train_loss)  
        if test_loss<minLoss:
            # keep the best model
            minLoss = test_loss
            bestmodel = model
    return model, bestmodel, Train_loss, Test_loss



# class data_loader():
#     """
#     TBD
    
#     reuse pytorch data load ??
    
#     code a "get" to get a batch size sample
#     """
    
#     def __init__(self, ds, Nhours, dt_forcing, batchsize):
#         """
#         """
        
def dataloader(
                arrays : dict, 
                batch_size: int
               ):
    first_key = list(arrays.keys())[0]
    dataset_size = arrays[first_key].shape[0] # <- time extent
    assert all(arrays[key].shape[0] == dataset_size for key in arrays.keys()) # <- check that all array are on the same time lenght
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            # yield tuple(array[batch_perm] for array in arrays)
            yield {key:array[batch_perm] for (key,array) in arrays.items() }
            start = end
            end = start + batch_size