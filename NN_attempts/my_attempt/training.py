import optax
import model
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Float, Int, Array
import jax.tree_util as jtu
from model import jslab

from tools import my_fc_filter

def my_partition(model):
    filter_spec = my_filter(model)
    return eqx.partition(model, filter_spec) 

def my_filter(model):
    filter_spec = jtu.tree_map(lambda tree: False, model) # keep nothing
    filter_spec = eqx.tree_at( lambda tree: tree.K0, filter_spec, replace=True)
    filter_spec = eqx.tree_at( lambda tree: tree.dissipation_model, filter_spec, replace=True)
    return filter_spec

# a loss
def loss(dynamic_model, static_model, data_obs):
    mymodel = eqx.combine(dynamic_model, static_model)
    nt = data_obs['U'].shape[0]
    t0 = 0.
    t1 = nt*mymodel.dt_forcing
    mymodel = eqx.tree_at( lambda t:t.t0, mymodel, t0)
    mymodel = eqx.tree_at( lambda t:t.t1, mymodel, t1)
    sol = mymodel()
    print(sol[0].shape, data_obs['U'].shape)
    
    jax.debug.print('nb of nan in sol U {}', jnp.sum(jnp.isnan(sol[0])))
    jax.debug.print('nb of nan in obs U {}', jnp.sum(jnp.isnan(data_obs['U'])))
    
    J = jnp.nanmean( (sol[0]-data_obs['U'])**2 + (sol[1]-data_obs['V'])**2 ) # , axis=(0,-1,-2)
    return J

def evaluate(model, test_data):
    dynamic_model, static_model = my_partition(model)
    nt = test_data['U'].shape[0]
    t0 = 0.
    t1 = nt*model.dt_forcing
    mymodel = eqx.tree_at( lambda t:t.t0, model, t0)
    mymodel = eqx.tree_at( lambda t:t.t1, mymodel, t1)
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
    
    #opt_state = optim.init( eqx.filter(model, eqx.is_array) )
    opt_state = optim.init( model )

    #print(eqx.filter(model, eqx.is_inexact_array))


    @eqx.filter_jit
    def make_step( model: jslab, opt_state: PyTree):
        
        dynamic_model, static_model = my_partition(model)
        
        jax.debug.print('before grad compute K0: {}', dynamic_model.K0)
        
        loss_value, grads = eqx.filter_value_and_grad(loss)(dynamic_model, static_model, train_data)
        
        jax.debug.print('after grad compute loss, K0: {}, {}', loss_value, grads.K0)
        
        
        
        updates, opt_state = optim.update( grads, opt_state, model) #,   dynamic_model )
        #jax.debug.print('loss, K0: {}, {}', loss_value, grads.K0)
        model = eqx.apply_updates(model, updates)
        # mymodel = eqx.combine(dynamic_model, static_model)
        # print(model)
        return model, opt_state, loss_value
    
    for step in range(maxstep):
        model, opt_state, train_loss = make_step(model, opt_state)
        #if (step % print_every) == 0 or (step == maxstep - 1):
        if (step % 1) == 0 or (step == maxstep - 1):
            test_loss = evaluate(model, test_data)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}"
            )
            
            
def dataset_maker(ds, Nhours):
    train_data = ds.isel(time=slice(0,-2*Nhours))
    test_data = ds.isel(time=slice(-2*Nhours,-Nhours))
    verif_data = ds.isel(time=slice(-Nhours,-1))
    
    ds_train, ds_test, ds_verif = {}, {}, {}
    for var in ['U','V']:
        ds_train[var] = jnp.asarray(train_data[var].values)
        ds_test[var] = jnp.asarray(test_data[var].values)
        ds_verif[var] = jnp.asarray(verif_data[var].values)
    return ds_train, ds_test, ds_verif


