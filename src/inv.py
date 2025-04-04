import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import equinox as eqx
import jax.tree_util as jtu
from jax import lax
import numpy as np

import scipy

import tools
from Listes_models import L_variable_Kt, L_nlayers_models, L_models_total_current

class Variational:
    """
    """
    def __init__(self, model, observations, filter_at_fc=False):
        self.observations = observations
        self.obs_period = observations.obs_period
        self.dt_forcing = model.dt_forcing
        self.model = model
        self.filter_at_fc = filter_at_fc

    def loss_fn(self, sol, obs):
        sol = jnp.asarray(sol)
        return jnp.nanmean( (sol[0]-obs[0])**2 + (sol[1]-obs[1])**2 )
    
    @eqx.filter_jit
    def cost(self, dynamic_model, static_model):
        mymodel = eqx.combine(dynamic_model, static_model)
        dtime_obs = self.observations.obs_period
        if type(mymodel).__name__ in L_models_total_current:
            utotal = True # Ut = Uag + Ug
        else:
            utotal = False # Uag
        obs = self.observations.get_obs(utotal)
        
        
        if self.filter_at_fc:
            dt_out = mymodel.dt_forcing
        else:
            dt_out = dtime_obs
        sol = mymodel(save_traj_at=dt_out)
        
        if type(mymodel).__name__ in L_nlayers_models:
            sol = (sol[0][:,0], sol[1][:,0]) 
        
        
        if self.filter_at_fc:
            
            # run the model at high frequency
            #Ua,Va = mymodel(save_traj_at=mymodel.dt_forcing)
            Ua, Va = sol
            Uf, Vf = tools.my_fc_filter(mymodel.dt_forcing, Ua + 1j*Va, mymodel.fc) # here filter at fc
            
            # lets now create an array of size 'obs', with the value from the filtered estimate
            Uffc = jnp.zeros(len(obs[0]))
            Vffc = jnp.zeros(len(obs[0]))
            step = jnp.array(self.obs_period//mymodel.dt_forcing,int)
            
            # for loop in JAX language  
            def _fn_for_scan(X0, k, Uf, Vf, step):
                Uffc,Vffc = X0
                Uffc = Uffc.at[k].set(Uf[k*step])  
                Vffc = Vffc.at[k].set(Vf[k*step]) 
                X0 = (Uffc,Vffc)
                return X0, X0
            final, _ = lax.scan(lambda X0, k:_fn_for_scan(X0, k, Uf, Vf, step), init=(Uffc,Vffc), xs=np.arange(0,len(Uffc)))     
            sol = final
        # else:
        #     sol = mymodel(save_traj_at=dtime_obs) # use diffrax and equinox 
            
        # if type(mymodel).__name__ in ['junsteak']:
        #     sol = (sol[0][:,0], sol[1][:,0]) # <- we observe first layer only
            
        return self.loss_fn(sol, obs)
        
   
        
            
    
    @eqx.filter_jit
    def grad_cost(self, dynamic_model, static_model):
        
        def cost_for_grad(dynamic_model, static_model):
            y = self.cost(dynamic_model, static_model)
            if static_model.AD_mode=='F': 
                return y,y # <- trick to have a similar behavior than value_and_grad (but here returns grad, value)
            else:
                return y
         
        if static_model.AD_mode=='F':
            val2, val1 =  eqx.filter_jacfwd(cost_for_grad, has_aux=True)(dynamic_model, static_model)
        else:
            val1, val2 = eqx.filter_value_and_grad(cost_for_grad)(dynamic_model, static_model)
    
        return val1, val2 

    
    def my_minimizer(self, opti, mymodel, itmax, gtol=1e-5, verbose=False):
        """
        wrapper of optax minimizer, updating 'model' as the loop goes
        """
        
        
        if opti=='adam':
            solver = optax.adam(1e-2)
            opt_state = solver.init(mymodel)
            
            @eqx.filter_jit
            def step_minimize(model, opt, opt_state):
                dynamic_model, static_model = my_partition(mymodel)
                value, grad = self.grad_cost(dynamic_model, static_model)
                value_grad = my_combine_params(grad) #  grad.pk 
                updates, opt_state = opt.update(grad, opt_state)
                model = eqx.apply_updates(model, updates)
                return value, value_grad, model, opt_state
            
        elif opti=='lbfgs':
            if mymodel.AD_mode=='F':
                raise Exception('Error: LBFGS in optax uses linesearch, that itself uses value_and_grad. You have chosen a forward automatic differentiation, exiting ...')
            else:
                """
                The rest of this minimizer, in reverse mode, is still WIP
                """
                
                linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=15,decrease_factor=0.9)
                solver = optax.scale_by_lbfgs() 
                solver = optax.chain( optax.scale_by_lbfgs(), linesearch)
                # Compare with or without linesearch by commenting this line
                
                        
                #opt_state = opt.init(value_and_grad_fn, mymodel.pk)
                opt_state = solver.init(mymodel.pk)
                
                
            
                def value_and_grad_fn(params): # L-BFGS expects a function that returns both value and gradient
                        dynamic_model, static_model = my_partition(mymodel)
                        #new_dynamic_model = my_replace(dynamic_model, params) # replace new pk
                        new_dynamic_model = eqx.tree_at(lambda tree:tree.pk, dynamic_model, params)
                        value, grad = self.grad_cost(new_dynamic_model, static_model)
                        return value, my_combine_params(grad) # grad.pk
                
                def cost_fn(params):
                    value, _ = value_and_grad_fn(params)
                    return value
                                
                def gstop_criterion(grad, gtol=1e-5):
                    return jnp.amax(jnp.abs(grad))>=gtol
                
                # optax.scale_by_zoom_linesearch() IS USING VALUE_AND_GRAD (SO REVERSE AD) !!!
                # l.1583 in optax/src/linesearch.py
                # lets use another linesearch : scale_by_backtracking_linesearch
                @eqx.filter_jit
                def step_minimize(carry):
                    value, grad, model, opt_state = carry
                    params = model.pk
                    
                    #dynamic_model, static_model = self.my_partition(model)
                    value, grad = value_and_grad_fn(params)
                    
                    updates, opt_state = solver.update(grad, opt_state, params,
                                                       value=value, grad=grad, value_fn=cost_fn) 
                    
                    params = optax.apply_updates(params, updates)
                                
                    # Apply updates to model
                    model = eqx.tree_at(lambda m: m.pk, model, updates)
                    #model = my_replace(model, updates)
                    return value, grad, model, opt_state


                # initialisation
                it = 0
                value, grad = value_and_grad_fn(mymodel.pk)
                
                # loop
                while it<itmax and gstop_criterion(grad, gtol): 
                    carry = value, grad, mymodel, opt_state
                    value, grad, mymodel, opt_state = step_minimize(carry)
                    if verbose:
                        print("it, J, K, |dJ|:",it, value, mymodel.pk, jnp.amax(grad))
                    it += 1
                    
                print('Final pk is:',mymodel.pk)
                return mymodel
    
    def scipy_lbfgs_wrapper(self, mymodel, maxiter, gtol=1e-5, verbose=False):
        
        # L-BFGS expects a function that returns both value and gradient
        @jax.jit
        def value_and_grad_for_scipy(params): 
            dynamic_model, static_model = my_partition(mymodel)
            #new_dynamic_model = my_replace(dynamic_model, params) # replace new params
            new_dynamic_model = eqx.tree_at( lambda tree:tree.pk, dynamic_model, params)
            value, grad = self.grad_cost(new_dynamic_model, static_model)
            return value, grad.pk
            

        vector_k = my_combine_params(mymodel) #mymodel.pk
        print(' intial pk',vector_k)
        
        res = scipy.optimize.minimize(value_and_grad_for_scipy, 
                                            vector_k,
                                            options={'maxiter':maxiter, 'gtol':gtol}, #
                                            method='L-BFGS-B',
                                            jac=True)
        
        
        
        new_k = jnp.asarray(res['x'])
        mymodel = eqx.tree_at( lambda tree:tree.pk, mymodel, new_k)
        #mymodel = my_replace(mymodel, new_k)
        if verbose:
            dynamic_model, static_model = my_partition(mymodel)
            value, gradtree = self.grad_cost(dynamic_model, static_model)
            grad = gradtree.pk
            print(res.message)
            print(' final cost, grad',value, grad)
            print('     vector K solution ('+str(res.nit)+' iterations)',res['x'])
            print('     cost function value with K solution:',value)
        
        return mymodel, res
    
def fn_where_pk(model):
    return jnp.array(model.pk)
def fn_where_dTK(model):
    return jnp.array(model.dTK)

list_fn = [fn_where_pk] #[fn_where_pk, fn_where_dTK]

def my_replace(model, newvalue):
    newmodel = model
    for fn in list_fn:
        newmodel = replace_leaf(fn, newmodel, newvalue)
    return newmodel
 
def my_combine_params(model):
    combined = list_fn[0](model)
    if len(list_fn)>1:
        for fn in list_fn:
            combined = jnp.append( combined, fn(model) )
    return combined       

@eqx.filter_jit
def my_partition(mymodel):
    filter_spec = jtu.tree_map(lambda arr: False, mymodel) # keep nothing
    filter_spec = eqx.tree_at( lambda tree: tree.pk, filter_spec, True) # keep pk  replace=
    # # filter_spec = eqx.tree_at( lambda tree: tree.dTK, filter_spec, True) # keep dTK
    # filter_spec = my_replace(filter_spec, True)
    return eqx.partition(mymodel, filter_spec)    

# tools manipulating pytrees
def replace_leaf(fn_where, model, newvalue):
    return eqx.tree_at( fn_where, model, newvalue)

