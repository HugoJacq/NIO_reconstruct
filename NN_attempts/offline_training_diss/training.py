import optax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree, Float, Int, Array
import jax.tree_util as jtu




def my_partition(model):
    filter_spec = my_filter(model)
    return eqx.partition(model, filter_spec) 

def my_filter(model):
    filter_spec = jtu.tree_map(lambda tree: False, model) # keep nothing
    #filter_spec = eqx.tree_at( lambda tree: tree.K, filter_spec, replace=True)
    filter_spec = eqx.tree_at( lambda tree: tree.dissipation_model, filter_spec, replace=True)
    return filter_spec

def loss(dynamic_model, static_model, data_obs):
    """
    TBD
    """

def evaluate(model, test_data):
    """
    TBD
    """

# train process
def train(
        model: eqx.Module,
        optim: optax.GradientTransformation,
        train_data: dict,
        test_data: dict,
        maxstep: Int,
        print_every: Int
            ):
    
    dynamic_model, static_model = my_partition(model)
    opt_state = optim.init( dynamic_model )

    @eqx.filter_jit
    def make_step( model, opt_state: PyTree):
        
        dynamic_model, static_model = my_partition(model)
        
        #jax.debug.print('before grad compute K0: {}', dynamic_model.K0)
        
        # BACKWARD AD
        print(dynamic_model)
        
        loss_value, grads = eqx.filter_value_and_grad(loss)(dynamic_model, static_model, train_data)
        
        #jax.debug.print('dK0: {}', grads.K0)
        # jax.debug.print('weight layer1: {}', grads.dissipation_model.layer1.weight)
        # jax.debug.print('bias layer1: {}', grads.dissipation_model.layer1.bias)
        # jax.debug.print('weight layer2: {}', grads.dissipation_model.layer2.weight)
        # jax.debug.print('bias layer2: {}', grads.dissipation_model.layer2.bias)
        
        # FORWARD AD
        # grads = eqx.filter_jacfwd(loss)(dynamic_model, static_model, train_data)
        # loss_value = loss(dynamic_model, static_model, train_data)
        
        #jax.debug.print('after grad compute loss, K0: {}, {}', loss_value, grads.K0)

        updates, opt_state = optim.update( grads, opt_state, model)
        #jax.debug.print('loss, K0: {}, {}', loss_value, grads.K0)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    for step in range(maxstep):
        model, opt_state, train_loss = make_step(model, opt_state)
        if (step % print_every) == 0 or (step == maxstep - 1):
            test_loss = evaluate(model, test_data)
            print(
                f"{step=}, train_loss={train_loss.item()}, " # train loss of current epoch (uses the old model)
                f"test_loss={test_loss.item()}" # test loss of next epoch (uses the new model)
            )
    return model



class data_loader():
    """
    TBD
    
    reuse pytorch data load ??
    
    code a "get" to get a batch size sample
    """
    
    def __init__(self, ds, Nhours, dt_forcing, batchsize):
        """
        """