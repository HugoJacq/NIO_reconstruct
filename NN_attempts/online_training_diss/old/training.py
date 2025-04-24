import optax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree, Float, Int, Array
import jax.tree_util as jtu

from model import jslab

def my_partition(model):
    filter_spec = my_filter(model)
    return eqx.partition(model, filter_spec) 

def my_filter(model):
    filter_spec = jtu.tree_map(lambda tree: False, model) # keep nothing
    #filter_spec = eqx.tree_at( lambda tree: tree.K0, filter_spec, replace=True)
    filter_spec = eqx.tree_at( lambda tree: tree.dissipation_model, filter_spec, replace=True)
    # filter_spec = eqx.tree_at( lambda tree: tree.dissipation_model.layer1.weight, filter_spec, replace=True)
    # filter_spec = eqx.tree_at( lambda tree: tree.dissipation_model.layer1.bias, filter_spec, replace=True)
    # filter_spec = eqx.tree_at( lambda tree: tree.dissipation_model.layer2.weight, filter_spec, replace=True)
    # filter_spec = eqx.tree_at( lambda tree: tree.dissipation_model.layer2.bias, filter_spec, replace=True)
    return filter_spec

# a loss
def loss(dynamic_model, static_model, data_obs):
    mymodel = eqx.combine(dynamic_model, static_model)
    t0, t1 = data_obs['t0'], data_obs['t1']
    sol = mymodel(t0,t1)    
    J = jnp.nanmean( (sol[0]-data_obs['U'])**2 + (sol[1]-data_obs['V'])**2 ) # , axis=(0,-1,-2)
    return J

def evaluate(model, test_data):
    dynamic_model, static_model = my_partition(model)
    avg_loss = jnp.mean(loss(dynamic_model, static_model, test_data))
    return avg_loss

# train process
def train(
        model: eqx.Module,
        optim: optax.GradientTransformation,
        train_data: dict,
        test_data: dict,
        maxstep: Int,
        print_every: Int
            ):
    
    dynamic_model, _ = my_partition(model)
    opt_state = optim.init( dynamic_model )

    @eqx.filter_jit
    def make_step( model: PyTree, opt_state: PyTree):
        
        dynamic_model, static_model = my_partition(model)
        
        #jax.debug.print('before grad compute K0: {}', dynamic_model.K0)
        
        # BACKWARD AD
        loss_value, grads = eqx.filter_value_and_grad(loss)(dynamic_model, static_model, train_data) # test avec jax.value_and_grad ?
        
        #jax.debug.print('dK0: {}', grads.K0)
        jax.debug.print('weight layer1: {}', grads.dissipation_model.layer1.weight)
        jax.debug.print('bias layer1: {}', grads.dissipation_model.layer1.bias)
        jax.debug.print('weight layer2: {}', grads.dissipation_model.layer2.weight)
        jax.debug.print('bias layer2: {}', grads.dissipation_model.layer2.bias)
        
        # FORWARD AD
        # grads = eqx.filter_jacfwd(loss)(dynamic_model, static_model, train_data)
        # loss_value = loss(dynamic_model, static_model, train_data)
        
        #jax.debug.print('after grad compute loss, K0: {}, {}', loss_value, grads.K0)

        updates, opt_state = optim.update( grads, opt_state, model)
        #jax.debug.print('loss, K0: {}, {}', loss_value, grads.K0)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    Test_loss = []
    Train_loss = []
    minLoss = 999999
    bestmodel = model
    for step in range(maxstep):
        model, opt_state, train_loss = make_step(model, opt_state)
        
        if (step % print_every) == 0 or (step == maxstep - 1):
            test_loss = evaluate(model, test_data)
            print(
                f"{step=}, train_loss={train_loss.item()}, " # train loss of current epoch (uses the old model)
                f"test_loss={test_loss.item()}" # test loss of next epoch (uses the new model)
            )
        Test_loss.append(test_loss)
        Train_loss.append(train_loss)  
        if test_loss<minLoss:
            # keep the best model
            minLoss = test_loss
            bestmodel = model
    
    return model, bestmodel, Train_loss, Test_loss
            

def dataset_maker(ds, Nhours, dt_forcing):
    train_data = ds.isel(time=slice(0,- Nhours))
    test_data = ds.isel(time=slice(- Nhours-1,-1))
    nt = len(ds.time)
    
    
    t0 = 0.
    tmid = (nt-Nhours)*dt_forcing
    t1 = nt*dt_forcing
    
    ds_train, ds_test = {}, {}
    for var in ['U','V']:
        ds_train[var] = jnp.asarray(train_data[var].values)
        ds_test[var] = jnp.asarray(test_data[var].values)
    ds_train['t0'] =  t0
    ds_train['t1'] = tmid
    ds_test['t0'] = tmid
    ds_test['t1'] = t1
    return ds_train, ds_test


