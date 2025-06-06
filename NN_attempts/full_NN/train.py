import optax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree, Float, Int, Array
import jax.tree_util as jtu


# a loss
def fn_loss(sol,obs):
    return jnp.nanmean( (sol[0]-obs[0])**2 + (sol[1]-obs[1])**2 )

def loss(NNmodel, batch_train, batch_obs):
    sol = jax.vmap(NNmodel, in_axes=1, out_axes=1)(batch_train)    # vmap is used to apply the model on every 'batch'
    return fn_loss(sol, batch_obs)

def evaluate(NNmodel, batch_val, batch_obs):
    avg_loss = jnp.mean(loss(NNmodel, batch_val, batch_obs))
    return avg_loss

def do_training(
        model: eqx.Module,
        optim: optax.GradientTransformation,
        features,
        obs,
        maxstep: Int,
        print_every: Int
            ):
    
    opt_state = optim.init( eqx.filter(model, eqx.is_array) ) # here, the filter is used to keep only arrays

    @eqx.filter_jit
    def make_step( NNmodel, opt_state: PyTree):
        
        loss_value, grads = eqx.filter_value_and_grad(loss)(NNmodel, features['train'], obs['train']) # BACKWARD AD        
        updates, opt_state = optim.update( grads, opt_state, NNmodel)
        NNmodel = eqx.apply_updates(NNmodel, updates)
        return NNmodel, opt_state, loss_value
    
    Test_loss = []
    Train_loss = []
    minLoss = 999
    for step in range(maxstep):
        model, opt_state, train_loss = make_step(model, opt_state)
        test_loss = evaluate(model, features['val'], obs['val'])
        if (step % print_every) == 0 or (step == maxstep - 1):
            #test_loss = evaluate(model, features['val'], obs['val'])
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}"
            )
        Test_loss.append(test_loss)
        Train_loss.append(train_loss)  
        if test_loss<minLoss:
            # keep the best model
            minLoss = test_loss
            bestmodel = model
    
    return model, bestmodel, Train_loss, Test_loss