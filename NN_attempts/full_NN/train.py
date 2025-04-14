import optax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree, Float, Int, Array
import jax.tree_util as jtu


# a loss
def loss(NNmodel, features_train, obs_train):
    sol = NNmodel(features_train)    
    J = jnp.nanmean( (sol[0]-obs_train[0])**2 + (sol[1]-obs_train[1])**2 )
    return J

def evaluate(NNmodel, features_val, obs_val):
    avg_loss = jnp.mean(loss(NNmodel, features_val, obs_val))
    return avg_loss

def do_training(
        model: eqx.Module,
        optim: optax.GradientTransformation,
        features,
        obs,
        maxstep: Int,
        print_every: Int
            ):
    
    opt_state = optim.init( model )

    @eqx.filter_jit
    def make_step( NNmodel, opt_state: PyTree):
        # BACKWARD AD
        loss_value, grads = eqx.filter_value_and_grad(loss)(NNmodel, features['train'], obs['train'])
        
        #jax.debug.print('dK0: {}', grads.K0)
        # jax.debug.print('weight layer1: {}', grads.dissipation_model.layer1.weight)
        # jax.debug.print('bias layer1: {}', grads.dissipation_model.layer1.bias)
        # jax.debug.print('weight layer2: {}', grads.dissipation_model.layer2.weight)
        # jax.debug.print('bias layer2: {}', grads.dissipation_model.layer2.bias)
        
        updates, opt_state = optim.update( grads, opt_state, NNmodel)
        NNmodel = eqx.apply_updates(NNmodel, updates)
        return NNmodel, opt_state, loss_value
    
    for step in range(maxstep):
        model, opt_state, train_loss = make_step(model, opt_state)
        if (step % print_every) == 0 or (step == maxstep - 1):
            test_loss = evaluate(model, features['val'], obs['val'])
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}"
            )
    return model