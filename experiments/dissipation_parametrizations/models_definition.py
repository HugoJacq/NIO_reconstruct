import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree
import jax
"""
* A RHS class that takes terms as input, each of them either a physical param or a NN (eqx.module)
"""


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
        return jnp.stack([self.K0*TA[0], self.K0*TA[1]])
    
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
        return jnp.stack([- self.R*C[0], -self.R*C[1]])
    
    def filter_set_trainable(self, filter_spec):
        return eqx.tree_at(lambda t: t.R, filter_spec, replace=True)
    
    
    
class Dissipation_CNN(eqx.Module):
    """
    """
    
    




####
# Utility functions
###

def replace_term(model, where, new_term):
    """"""