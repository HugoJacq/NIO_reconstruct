import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import equinox as eqx
import jax.tree_util as jtu

import scipy



class Variational_diffrax:
    """
    """
    def __init__(self, model, observations, is_difx=True):
        self.observations = observations
        self.obs_period = observations.obs_period
        self.dt_forcing = model.dt_forcing
        self.model = model
        self.is_difx = is_difx

    def loss_fn(self, obs, sol):
        return jnp.mean( (sol[0]-obs[0])**2 + (sol[1]-obs[1])**2 )
    
    def cost(self, dynamic_model, static_model, call_args):
        mymodel = eqx.combine(dynamic_model, static_model)
        dtime_obs = self.observations.obs_period
        obs = self.observations.get_obs()
        if self.is_difx:
            sol = mymodel(call_args, save_traj_at=dtime_obs).ys # use diffrax and equinox
        else:
            sol = mymodel(call_args, save_traj_at=dtime_obs) # only equinox
        return self.loss_fn(sol, obs)
        
   
        
    def my_partition(self, mymodel):
        filter_spec = jtu.tree_map(lambda arr: False, mymodel) # keep nothing
        filter_spec = eqx.tree_at( lambda tree: tree.pk, filter_spec, replace=True) # keep only pk
        return eqx.partition(mymodel, filter_spec)          
    
    @eqx.filter_jit
    def grad_cost(self, dynamic_model, static_model, call_args):
        def cost_for_grad(dynamic_model, static_model, call_args):
            y = self.cost(dynamic_model, static_model, call_args)
            if static_model.AD_mode=='F':
                return y,y # <- trick to have a similar behavior than value_and_grad (but here returns grad, value)
            else:
                return y
        
        if self.model.AD_mode=='F':
            val1, val2 =  eqx.filter_jacfwd(cost_for_grad, has_aux=True)(dynamic_model, static_model, call_args)
            return val2, val1
        else:
            val1, val2 = eqx.filter_value_and_grad(cost_for_grad)(dynamic_model, static_model, call_args)
            return val1, val2
    
    
    

    def my_minimizer(self, opti, mymodel, itmax, call_args, gtol=1e-5, verbose=False):
        """
        wrapper of optax minimizer, updating 'model' as the loop goes
        """
        
        
        if opti=='adam':
            solver = optax.adam(1e-2)
            opt_state = solver.init(mymodel)
            
            @eqx.filter_jit
            def step_minimize(model, opt, opt_state, call_args):
                dynamic_model, static_model = self.my_partition(mymodel)
                value, grad = self.grad_cost(dynamic_model, static_model, call_args)
                value_grad = grad.pk 
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
                        dynamic_model, static_model = self.my_partition(mymodel)
                        new_dynamic_model = eqx.tree_at(lambda m: m.pk, dynamic_model, params) # replace new pk
                        value, grad = self.grad_cost(new_dynamic_model, static_model, call_args)
                        return value, grad.pk
                
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
    
    def scipy_lbfgs_wrapper(self, mymodel, maxiter, call_args, verbose=False):
        
        # L-BFGS expects a function that returns both value and gradient
        def value_and_grad_for_scipy(params): 
                    dynamic_model, static_model = self.my_partition(mymodel)
                    new_dynamic_model = eqx.tree_at(lambda m: m.pk, dynamic_model, params) # replace new pk
                    value, grad = self.grad_cost(new_dynamic_model, static_model, call_args)
                    return value, grad.pk
            

        vector_k = mymodel.pk
        print('intial pk',vector_k)
        
        res = scipy.optimize.minimize(value_and_grad_for_scipy, 
                                            vector_k,
                                            options={'maxiter':maxiter},
                                            method='L-BFGS-B',
                                            jac=True)
        
        new_k = jnp.asarray(res['x'])
        mymodel = eqx.tree_at( lambda tree:tree.pk, mymodel, new_k)

        if verbose:
            dynamic_model, static_model = self.my_partition(mymodel)
            value, gradtree = self.grad_cost(dynamic_model, static_model, call_args)
            grad = gradtree.pk
            print(' final cost, grad',value, grad)
            print('     vector K solution ('+str(res.nit)+' iterations)',res['x'])
            print('     cost function value with K solution:',value)
        
        return mymodel