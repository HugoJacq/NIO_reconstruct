from functools import partial
from tabnanny import verbose
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
  
  
def fn_loss(sol, obs, use_amplitude=False):
    # return jnp.mean( safe_for_grad_sqrt( (sol[:,0]-obs[:,0])**2 + (sol[:,1]-obs[:,1])**2 ))
    J = lax.select(use_amplitude, 
                    safe_for_grad_sqrt( (get_amplitude(sol[0],sol[1]) - get_amplitude(obs[0],obs[1]))**2 ),
                    safe_for_grad_sqrt( (sol[0]-obs[0])**2 + (sol[1]-obs[1])**2 ) )
    return jnp.nanmean( J )
    
def loss(dynamic_model, static_model, forcing, features, target, N_integration_steps, dt, dt_forcing, norms_features, use_amplitude):
    """
    Compute the distance between target and the integrated trajectory. 
    if N_integration_steps==1, estimates only the next time currents, otherwise compares trajectories
    """
    RHS_model = eqx.combine(dynamic_model, static_model)
    
    X0 = forcing[0,0:2,:,:]
    TA = forcing[:,2:4,:,:]
    
    # integration
    sol = Integration_Euler(X0, TA, features, RHS_model, dt, dt_forcing, N_integration_steps, norms_features)  
    return fn_loss(sol, target, use_amplitude)  
    
# @partial( jax.jit, static_argnames=['N_integration_steps','dt','dt_forcing'])
def vmap_loss(dynamic_model, static_model, data_batch, N_integration_steps, dt, dt_forcing, norms_features, use_amplitude):
    """
    vmap-like on a set of initial conditions, generates a number of trajectories (batch_size//N_integration_steps)
        at most: batch_size trajectories (each of length 1), at less: 1 trajectory (of length batch_size)
    
    Note: it has been assured beforehand that
                batch_size % N_integration_steps == 0
    """
    batch_size, _, Ny, Nx = data_batch['target'].shape
    Ntraj = batch_size//N_integration_steps    
    
    def fn_for_scan(L, k):
        start = k*N_integration_steps
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
                        norms_features,
                        use_amplitude)
        
        # jax.debug.print('loss_value {}',loss_value)
        
        L = L + loss_value
        return L, None
        
    L = 0.
    L, _ = eqx.internal.scan(fn_for_scan, L, jnp.arange(0,Ntraj), kind='checkpointed')

    return L / Ntraj
    
    
    
    
def train(the_model          : eqx.Module,
        optim               : optax.GradientTransformation,
        iter_train_data     : dict,
        test_data           : dict,
        maxstep             : int = 500,
        print_every         : int = 10,
        tol                 = None,
        retol               : float = 0.001,
        N_integration_steps : int = 2, # default is offline mode
        dt                  : float = 60.,
        dt_forcing          : float = 3600.,
        L_to_be_normalized  : list = [],
        use_amplitude       : bool = False,
        # verbose             : bool = False
            ):
    """
    Train loop to estimate parameters from 'the_model'.
    
    INPUTS:
        - the_model         : A model written in JAX, using equinox Modules
        - optim             : optimizer from Optax
        - iter_train_data   : a generator that loop through a dataset given a batch size
        - test_data         : a test data set, independent of train data set
        - maxstep           : number of max iteration
        - print_every       : show train_loss, test_loss every 'print_every'
        - tol               : threshold on max of gradients. If None, not stoping
        - retol            : threshold for relavive improvment of cost function
        - N_integration_steps: length of time integration (number of 'dt_forcing')
                            N=1 is offline mode, N>1 is online mode.
        - dt                : time step of model (inner loop)
        - dt_forcing        : time step of forcing (model's outer loop)
        - L_to_be_normalized: a list of term from ['forcing','target','features']
    
    OUTPUTS:
        a tuple with:
            - lastmodel     : the last iteration model
            - bestmodel     : the best model (lowest test loss)
            - Train_loss    : list of value of train loss
            - Test_loss     : list of value of test loss
            - opt_state_save : optimizer state for saving
    """
    
    @eqx.filter_jit
    def make_step( model, train_batch, opt_state):
        loss_value, grads = jax.value_and_grad(vmap_loss)(model, 
                                                          static_model, 
                                                          train_batch,
                                                          N_integration_steps,
                                                          dt,
                                                          dt_forcing, 
                                                          norms,
                                                          use_amplitude)        
        
        # loss_value = vmap_loss(model, static_model, train_batch, N_integration_steps, #  <- forward AD
        #                         dt, dt_forcing, norms, use_amplitude)
        # grads = jax.jacfwd(vmap_loss)(model, static_model, train_batch, N_integration_steps,
        #                         dt, dt_forcing, norms, use_amplitude)
        updates, opt_state = optim.update(grads, opt_state, model,
                                          value=loss_value,
                                          grad=grads,
                                          value_fn=lambda d:vmap_loss(d, static_model, 
                                                                    train_batch,
                                                                    N_integration_steps,
                                                                    dt,
                                                                    dt_forcing, 
                                                                    norms,
                                                                    use_amplitude)
                                          )
        new_dyn = eqx.apply_updates(model, updates)
        # jax.debug.print('New K0 {}, new R {}', new_dyn.stress_term.K0, new_dyn.dissipation_term.R)
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
                                 N_integration_steps=test_data['target'].shape[0],
                                 dt=dt,
                                 dt_forcing=dt_forcing, 
                                 norms_features=test_norms,
                                 use_amplitude=use_amplitude)
            print(
                f"{step=}, train_loss={train_loss.item()}, "    # train loss of current epoch (uses the old model)
                f"test_loss={test_loss.item()}"                 # test loss of next epoch (uses the new model)
            )
            Test_loss.append(test_loss.item())
        Train_loss.append(train_loss.item())  
        
        ###################
        # SAVING BEST MODEL
        ###################
        if test_loss<minLoss:
            # keep the best model
            minLoss = test_loss
            bestdyn = dynamic_model 
            opt_state_save = opt_state   

        ###################
        # Exit if converged
        ###################
        
        grads_array_only = eqx.filter(grads, eqx.is_array, replace=0.)
        maxgrad = optax.tree_utils.tree_max(jtu.tree_map(jnp.abs, grads_array_only))
        print(f'    max of grad is {maxgrad}')
        
        if tol is not None:
            # leafs, _ = jtu.tree_flatten(grads, is_leaf=eqx.is_array)
            # maxgrad = jnp.amax(jnp.abs(jnp.asarray(leafs)))
            # print(f'    max of grad is {maxgrad}')
             # print(f' list of grad is : {leafs}')

            if maxgrad < tol:
                CONVERGED = True
                print(f' loop has converged with max(grads)={maxgrad} < tol={tol}')
                # print(f' list of grad is : {leafs}')
        
        if retol is not None and step>1:
            rel_decrease = (Train_loss[-1] - Train_loss[-2])/Train_loss[-2]
            if rel_decrease < 0. and np.abs(rel_decrease)<retol:
                CONVERGED = True
                print(f'    loop has converged with relative decrease of L = {np.abs(rel_decrease)} < retol={retol} ')
                
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
    
@jax.jit
def safe_for_grad_sqrt(x):
  y = jnp.sqrt(jnp.where(x != 0., x, 1.))  # replace x=0. with any non zero real
  return jnp.where(x != 0., y, 0.)  # replace it back with O. (or x)

def get_amplitude(U,V):
    return safe_for_grad_sqrt(U**2+V**2)