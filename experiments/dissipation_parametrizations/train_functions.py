import jax
import equinox as eqx
import optax
from jaxtyping import PyTree
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import lax
import numpy as np


from train_preprocessing import normalize_batch
from time_integration import Integration_Euler    
  
  
def fn_loss(sol, obs):
    return jnp.mean( safe_for_grad_sqrt( (sol[:,0]-obs[:,0])**2 + (sol[:,1]-obs[:,1])**2 ))
    
def loss(dynamic_model, static_model, forcing, features, target, N_integration_steps, dt, dt_forcing, norms_features):
    """
    Compute the distance between target and the integrated trajectory. 
    if N_integration_steps==1, estimates only the next time currents, otherwise compares trajectories
    """
    RHS_model = eqx.combine(dynamic_model, static_model)
    
    X0 = forcing[0,0:2,:,:]
    TA = forcing[:,2:4,:,:]
    
    # integration
    sol = Integration_Euler(X0, TA, features, RHS_model, dt, dt_forcing, N_integration_steps, norms_features)  
    return fn_loss(sol, target)  
    
def vmap_loss(dynamic_model, static_model, data_batch, N_integration_steps, dt, dt_forcing, norms_features):
    """
    vmap on the initial condition, generates a number of trajectories (batch_size//N_integration_steps)
    at most: batch_size trajectories (each of length 1), at less: 1 trajectory (of length batch_size)
    
    Note: it has been assured beforehand that
                batch_size % N_integration_steps == 0
    """
    batch_size, _, Ny, Nx = data_batch['target'].shape
    Ntraj = batch_size//N_integration_steps    
    
    def fn_for_scan(L, k):
        start = k*N_integration_steps
        # print(data_batch['forcing'].shape, (N_integration_steps+1, data_batch['forcing'].shape[1], Ny, Nx))
        forcing_for_this_traj = lax.dynamic_slice(data_batch['forcing'],    (start, 0, 0, 0),   (N_integration_steps, data_batch['forcing'].shape[1], Ny, Nx))
        features_for_this_traj = lax.dynamic_slice(data_batch['features'],  (start, 0, 0, 0),   (N_integration_steps, data_batch['features'].shape[1], Ny, Nx))
        target_for_this_traj = lax.dynamic_slice(data_batch['target'],      (start, 0, 0, 0),   (N_integration_steps, data_batch['target'].shape[1], Ny, Nx))

        loss_value = loss(dynamic_model, 
                        static_model, 
                        forcing_for_this_traj,
                        features_for_this_traj,
                        target_for_this_traj,
                        N_integration_steps,
                        dt,
                        dt_forcing, 
                        norms_features)
        
        # jax.debug.print('loss_value {}',loss_value)
        
        L = L + loss_value
        return L, None
        
    L = 0.
    L, _ = lax.scan(fn_for_scan, L, jnp.arange(0,Ntraj))

    return L / Ntraj
    
    
    
    
def train(the_model          : eqx.Module,
        optim               : optax.GradientTransformation,
        iter_train_data     : dict,
        test_data           : dict,
        maxstep             : int = 500,
        print_every         : int = 10,
        save_best_model     : bool = True,
        tol                 = None,
        N_integration_steps : int = 1, # default is offline mode
        dt                  : float = 60.,
        dt_forcing          : float = 3600.,
        L_to_be_normalized  : list = []
            ):
    
    @eqx.filter_jit
    def make_step( model, train_batch, opt_state):
        loss_value, grads = jax.value_and_grad(vmap_loss)(model, 
                                                          static_model, 
                                                          train_batch,
                                                          N_integration_steps,
                                                          dt,
                                                          dt_forcing, 
                                                          norms)
        # loss_value = vmap_loss(model, static_model, train_batch, N_integration_steps,
        #                         dt, dt_forcing, norms)
        # grads = jax.jacfwd(vmap_loss)(model, static_model, train_batch, N_integration_steps,
        #                         dt, dt_forcing, norms)
        updates, opt_state = optim.update(grads, opt_state, model,
                                          value=loss_value,
                                          grad=grads,
                                          value_fn=lambda d:vmap_loss(d, static_model, 
                                                                    train_batch,
                                                                    N_integration_steps,
                                                                    dt,
                                                                    dt_forcing, 
                                                                    norms)
                                          )
        new_dyn = eqx.apply_updates(model, updates)
        jax.debug.print('New K0 {}, new R {}', new_dyn.stress_term.K0, new_dyn.dissipation_term.R)
        return new_dyn, opt_state, loss_value, grads
        
    # =================
    # INITIALIZATION
    # =================
    dynamic_model, static_model = my_partition(the_model)
    opt_state = optim.init(dynamic_model)
    
    Test_loss = []
    Train_loss = []
    minLoss = 999
        
    bestdyn = dynamic_model
    opt_state_save = opt_state
    evaluate = vmap_loss
    CONVERGED = False
    
    # =================
    # TRAIN LOOP
    # =================
    for step, batch_data in zip(range(maxstep), iter_train_data):    
        print('')
        ####################################
        # NORMALIZATION AT BATCH (NN inputs)
        ####################################
        batch_data, norms = normalize_batch(batch_data, L_to_be_normalized) # <- modify in place 'batch_data'

        ###################
        # MAKE STEP
        ###################
        dynamic_model, opt_state, train_loss, grads = make_step(dynamic_model, batch_data, opt_state)

        ###################
        # LOSS PLOT DATA
        ###################
        if (step % print_every) == 0 or (step == maxstep - 1):
            test_data, test_norms = normalize_batch(test_data, L_to_be_normalized) # normalize test features
            test_loss = evaluate(dynamic_model, 
                                 static_model, 
                                 test_data,
                                 N_integration_steps,
                                 dt,
                                 dt_forcing, 
                                 test_norms)
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

        ###################
        # Exit if converged
        ###################
        if tol is not None:
            leafs, _ = jtu.tree_flatten(grads, is_leaf=eqx.is_array)
            # print(f' list of grad is : {leafs}')
            maxgrad = jnp.amax(jnp.abs(jnp.asarray(leafs)))
            if maxgrad < tol:
                CONVERGED = True
                print(f' loop has converged with tol={tol}')
            print(f' list of grad is : {leafs}')
                
        if CONVERGED or step==maxstep-1:        
            lastmodel = eqx.combine(dynamic_model, static_model)
            bestmodel = eqx.combine(bestdyn, static_model)    
            break
        
    return lastmodel, bestmodel, Train_loss, Test_loss, opt_state_save
    
    
def my_partition(model):
    """
    This function splits the 'model' into a trainable part and static part.
    In the trainable model, all static arrays are set to 'None',
    while in the static model, all trainable arrays are set to 'None'.
    """
    filter_spec = jtu.tree_map(lambda x: False, model)    
    for term_name, term in model.__dict__.items():
        if (term_name[0:2]!='__') and (term.to_train):
            filter_for_term = term.filter_set_trainable(filter_spec.__dict__[term_name])
            filter_spec = eqx.tree_at( lambda t:t.__dict__[term_name], filter_spec, filter_for_term)
    return eqx.partition(model, filter_spec) 
    
    
def safe_for_grad_sqrt(x):
  y = jnp.sqrt(jnp.where(x != 0., x, 1.))  # replace x=0. with any non zero real
  return jnp.where(x != 0., y, 0.)  # replace it back with O. (or x)