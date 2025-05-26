import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree
import jax
import numpy as np
import jax.tree_util as jtu
"""
* A RHS class that takes terms as input, each of them either a physical param or a NN (eqx.module)
"""
from train_functions import my_partition

class RHS(eqx.Module):
    
    coriolis_term : PyTree
    stress_term : PyTree
    dissipation_term : PyTree

    def __init__(self, coriolis_term, stress_term, dissipation_term):
        self.coriolis_term = coriolis_term
        self.stress_term = stress_term
        self.dissipation_term = dissipation_term

    def __call__(self, C, TA, features):
        return self.coriolis_term(C) + self.stress_term(TA) + self.dissipation_term(features)


class Stress_term(eqx.Module):
    """
    K0.Tau with K0=1/H, Tau wind stress at surface stacked
    
    Inputs are TAx and TAy stacked on first dimension
    """
    K0 : Array = eqx.field(converter=jax.numpy.asarray) 
    to_train : bool
    
    def __init__(self, K0, to_train=False):
        self.K0 = K0
        self.to_train = to_train
        
    def __call__(self, TA: Float[Array, "2 Ny Nx"]) -> Float[Array, "2 Ny Nx"]:
        return jnp.stack([jnp.exp(self.K0)*TA[0], jnp.exp(self.K0)*TA[1]])
    
    def filter_set_trainable(self, filter_spec):
        return eqx.tree_at(lambda t: t.K0, filter_spec, replace=True)
     
    
class Coriolis_term(eqx.Module):
    """
    -i.fc.C with fc Coriolis frequency, C currents stacked
    
    Inputs are U and V stacked on first dimension
    """
    fc : Array = eqx.field(converter=jax.numpy.asarray) 
    to_train : bool
    
    def __init__(self, fc, to_train=False):
        self.fc = fc
        self.to_train = to_train
        
    def __call__(self, C: Float[Array, "2 Ny Nx"]) -> Float[Array, "2 Ny Nx"]:
        return jnp.stack([self.fc*C[1], -self.fc*C[0]])
    
    def filter_set_trainable(self, filter_spec):
        return filter_spec # nothing is trained here, so this function is identity
    
class Dissipation_Rayleigh(eqx.Module):
    """
    D = -rC with r real, C currents stacked
    
    Inputs are U and V stacked on first dimension
    """
    R : Array = eqx.field(converter=jax.numpy.asarray) 
    to_train : bool
    
    def __init__(self, R, to_train=False):
        self.R = R
        self.to_train = to_train
        
    def __call__(self, C: Float[Array, "2 Ny Nx"]) -> Float[Array, "2 Ny Nx"]:
        return jnp.stack([- jnp.exp(self.R)*C[0], -jnp.exp(self.R)*C[1]])
    
    def filter_set_trainable(self, filter_spec):
        return eqx.tree_at(lambda t: t.R, filter_spec, replace=True)
    
          
class Dissipation_CNN(eqx.Module):
    """
    Dissipation is parametrized as a CNN.
    
    Inputs are features normalized
    """
    layers: list
    to_train : bool
    
    # renormalization values (mean,std)
    NORMmean  : np.array 
    NORMstd    : np.array
    
    def __init__(self, key, Nfeatures, dtype='float32', to_train=True):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
                
        self.layers = [eqx.nn.Conv2d(Nfeatures, 8, padding='SAME', kernel_size=3, key=key1),
                        jax.nn.relu,
                        eqx.nn.Conv2d(8, 4, padding='SAME', kernel_size=3, key=key2),
                        eqx.nn.MaxPool2d(kernel_size=3),
                        eqx.nn.Conv2d(4, 2, padding='SAME', kernel_size=3, key=key3),
                        eqx.nn.MaxPool2d(kernel_size=3),
                        jnp.ravel,
                        eqx.nn.Linear(2*124*124, 256, key=key4),
                        eqx.nn.Linear(256, 2*128*128, key=key5),
                        ]
        # intialization of last linear layers
        # weights as normal distribution, std=1. and mean=0.
        # bias as 0.
        alpha = 1e-4
        initializer = jax.nn.initializers.normal(stddev=alpha) # mean is 0.
        self.layers[-1] = eqx.tree_at( lambda t:t.weight, self.layers[-1], initializer(key5, (2*128*128,256 )))
        self.layers[-1] = eqx.tree_at( lambda t:t.bias, self.layers[-1], self.layers[-1].bias*0.)
        
        
        self.NORMmean, self.NORMstd = np.zeros(2, dtype=dtype), np.ones(2, dtype=dtype)   
        self.to_train = to_train
        
    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        for layer in self.layers:
            x = layer(x)      
        x = jnp.reshape(x, (2,128,128))
        return x*1e-2
    
    def filter_set_trainable(self, filter_spec):
        for i, layer in enumerate(filter_spec.layers):
            if isinstance(layer, eqx.nn.Linear) or isinstance(layer, eqx.nn.Conv):
                filter_spec = eqx.tree_at(lambda t: t.layers[i].weight, filter_spec, replace=True)
                filter_spec = eqx.tree_at(lambda t: t.layers[i].bias, filter_spec, replace=True)
        return filter_spec
    

class Dissipation_MLP(eqx.Module):
    """
    Dissipation is parametrized as a MLP.
    
    Inputs are features normalized
    """
    layers: list
    to_train : bool
    
    # renormalization values (mean,std)
    NORMmean  : np.array 
    NORMstd    : np.array
    
    def __init__(self, key, Nfeatures, dtype='float32', to_train=True):
        key1, key2, key3 = jax.random.split(key, 3)
        depth = 64
        self.layers = [jnp.ravel,
                        eqx.nn.Linear(Nfeatures*128*128, depth, key=key1),
                        jax.nn.tanh,
                        eqx.nn.Linear(depth, depth, key=key2),
                        jax.nn.tanh,
                        eqx.nn.Linear(depth, 2*128*128, key=key3),
                        ]  
        # intialization of last linear layers
        # weights as normal distribution, std=1. and mean=0.
        # bias as 0.
        alpha = 1e-4
        initializer = jax.nn.initializers.normal(stddev=alpha) # mean is 0.
        self.layers[-1] = eqx.tree_at( lambda t:t.weight, self.layers[-1], initializer(key3, (2*128*128, depth)))
        self.layers[-1] = eqx.tree_at( lambda t:t.bias, self.layers[-1], self.layers[-1].bias*0.)
         
        self.NORMmean, self.NORMstd = np.zeros(2, dtype=dtype), np.ones(2, dtype=dtype)     
        self.to_train = to_train
        
    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        for layer in self.layers:
            x = layer(x)    
        x = jnp.reshape(x, (2,128,128)) 
        return x 

    def filter_set_trainable(self, filter_spec):
        for i, layer in enumerate(filter_spec.layers):
            if isinstance(layer, eqx.nn.Linear) or isinstance(layer, eqx.nn.Conv):
                filter_spec = eqx.tree_at(lambda t: t.layers[i].weight, filter_spec, replace=True)
                filter_spec = eqx.tree_at(lambda t: t.layers[i].bias, filter_spec, replace=True)
        return filter_spec
    
    
class Dissipation_MLP_linear(eqx.Module):
    """
    Dissipation is parametrized as a MLP, no activation function
    
    Inputs are features normalized
    """
    layers: list
    to_train : bool
    
    # renormalization values (mean,std)
    NORMmean  : np.array 
    NORMstd    : np.array
    
    def __init__(self, key, Nfeatures, dtype='float32', to_train=True):
        key1, key2, key3 = jax.random.split(key, 3)
        depth = 64
        self.layers = [jnp.ravel,
                        eqx.nn.Linear(Nfeatures*128*128, depth, key=key1),
                        eqx.nn.Linear(depth, depth, key=key2),
                        eqx.nn.Linear(depth, 2*128*128, key=key3),
                        ]  
        # intialization of last linear layers
        # weights as normal distribution, std=1. and mean=0.
        # bias as 0.
        alpha = 1e-5
        initializer = jax.nn.initializers.normal(stddev=alpha) # mean is 0.
        self.layers[-1] = eqx.tree_at( lambda t:t.weight, self.layers[-1], initializer(key3, (2*128*128, depth)))
        self.layers[-1] = eqx.tree_at( lambda t:t.bias, self.layers[-1], self.layers[-1].bias*0.)
         
        self.NORMmean, self.NORMstd = np.zeros(2, dtype=dtype), np.ones(2, dtype=dtype)     
        self.to_train = to_train
        
    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        for layer in self.layers:
            x = layer(x)    
        x = jnp.reshape(x, (2,128,128)) 
        return x 

    def filter_set_trainable(self, filter_spec):
        for i, layer in enumerate(filter_spec.layers):
            if isinstance(layer, eqx.nn.Linear) or isinstance(layer, eqx.nn.Conv):
                filter_spec = eqx.tree_at(lambda t: t.layers[i].weight, filter_spec, replace=True)
                filter_spec = eqx.tree_at(lambda t: t.layers[i].bias, filter_spec, replace=True)
        return filter_spec

####################
# Utility functions
####################
def get_number_of_param(model: eqx.Module):
    filtered,_ = my_partition(model)
    leafs, _ = jtu.tree_flatten(filtered)
    S = 0
    for k in range(len(leafs)):
        if eqx.is_array(leafs[k]):
            S = S+ leafs[k].size
    return S
