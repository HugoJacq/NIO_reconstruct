import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from functools import partial
from typing import Tuple, Dict, Any


class PhysicsInformedSolver(eqx.Module):
    model: eqx.Module
    K0: jnp.ndarray
    dt: float
    
    def __init__(self, dissipation_model, dt=3600.0, key=None):  # dt in seconds (default 1 hour)
        self.model = dissipation_model
        self.dt = dt
        # Learnable parameter for wind stress coefficient
        self.K0 = jnp.array([0.1], dtype=jnp.float32)
    
    def coriolis_term(self, U, f):
        """Calculate -i*f*U term (Coriolis)"""
        # U is complex, represented as [real, imag]
        real, imag = U[:, 0], U[:, 1]
        return jnp.stack([-f * imag, f * real], axis=1)
    
    def wind_term(self, tau):
        """Calculate K0*tau term"""
        return self.K0 * tau
    
    def rhs(self, U, features, tau, f, u_mag):
        """Right-hand side of the PDE"""
        coriolis = self.coriolis_term(U, f)
        wind = self.wind_term(tau)
        
        # Get dissipation from neural network (ensuring it's positive)
        dissipation = self.model(features, u_mag)
        
        # Convert dissipation to complex form aligned with velocity direction
        dissip_real = dissipation * U[:, 0] / (u_mag + 1e-8)
        dissip_imag = dissipation * U[:, 1] / (u_mag + 1e-8)
        dissipation_term = jnp.stack([dissip_real, dissip_imag], axis=1)
        
        # Complete right-hand side
        return coriolis + wind - dissipation_term
    
    def rk4_step(self, U, features, tau, f):
        """4th order Runge-Kutta integration step"""
        u_mag = jnp.sqrt(U[:, 0]**2 + U[:, 1]**2)
        
        # RK4 steps
        k1 = self.rhs(U, features, tau, f, u_mag)
        k2 = self.rhs(U + 0.5 * self.dt * k1, features, tau, f, u_mag)
        k3 = self.rhs(U + 0.5 * self.dt * k2, features, tau, f, u_mag)
        k4 = self.rhs(U + self.dt * k3, features, tau, f, u_mag)
        
        # Update
        U_next = U + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return U_next
    
    def compute_losses(self, U_pred, U_obs, residual):
        """Compute physics-informed loss function"""
        # Data matching loss
        data_loss = jnp.mean((U_pred - U_obs)**2)
        
        # Physics residual loss
        physics_loss = jnp.mean(residual**2)
        
        # Regularization on K0 to keep it physical
        k0_reg = jnp.abs(self.K0 - 0.15)**2  # Prior knowledge that K0 ≈ 0.15
        
        # Weighted loss
        total_loss = data_loss + 0.1 * physics_loss + 0.01 * k0_reg
        
        return total_loss, {
            'data': data_loss.item(),
            'physics': physics_loss.item(),
            'k0_reg': k0_reg.item()
        }


@partial(jax.jit, static_argnums=(3,))
def train_step(solver, opt_state, batch, optimizer):
    U_t, U_t_plus_dt, features, tau, f = batch
    
    # Define loss function
    def loss_fn(solver):
        # Forward simulation
        U_pred = solver.rk4_step(U_t, features, tau, f)
        
        # Calculate physics residual
        u_mag = jnp.sqrt(U_t[:, 0]**2 + U_t[:, 1]**2)
        du_dt = (U_t_plus_dt - U_t) / solver.dt
        rhs = solver.rhs(U_t, features, tau, f, u_mag)
        residual = du_dt - rhs
        
        # Calculate losses
        loss, loss_components = solver.compute_losses(U_pred, U_t_plus_dt, residual)
        return loss, (loss_components, U_pred)
    
    # Compute gradients
    (loss, (loss_components, U_pred)), grads = jax.value_and_grad(loss_fn, has_aux=True)(solver)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    solver = eqx.apply_updates(solver, updates)
    
    return solver, opt_state, loss, loss_components, U_pred


def create_train_loop(solver, learning_rate=1e-3):
    # Split model into static and trainable parts
    solver_trainable, solver_static = eqx.partition(
        solver, lambda x: eqx.is_array(x) and not (isinstance(x, jnp.ndarray) and x.size == 0)
    )
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(solver_trainable)
    
    # Create JIT-compiled training step
    @partial(jax.jit, static_argnums=(0,))
    def step(optimizer, solver, opt_state, batch):
        solver_trainable, solver_static = eqx.partition(
            solver, lambda x: eqx.is_array(x) and not (isinstance(x, jnp.ndarray) and x.size == 0)
        )
        
        solver_trainable, opt_state, loss, loss_components, U_pred = train_step(
            eqx.combine(solver_trainable, solver_static),
            opt_state,
            batch,
            optimizer
        )
        
        solver = eqx.combine(solver_trainable, solver_static)
        return solver, opt_state, loss, loss_components, U_pred
    
    return step, opt_state


# Example usage
def train(solver, train_data, num_epochs=100, batch_size=32, learning_rate=1e-3):
    step_fn, opt_state = create_train_loop(solver, learning_rate)
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch in create_batches(train_data, batch_size):
            solver, opt_state, loss, loss_components, _ = step_fn(
                optax.adam(learning_rate), solver, opt_state, batch
            )
            epoch_losses.append(loss)
        
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        print(f"Epoch {epoch}, Loss: {avg_loss}")
        
        # Extract K0 value for monitoring
        K0_value = solver.K0.item()
        print(f"Current K0 value: {K0_value}")
    
    return solver


def create_batches(data, batch_size):
    # Simple batch creation - in practice you'd shuffle, etc.
    U_t, U_t_plus_dt, features, tau, f = data
    num_samples = U_t.shape[0]
    
    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        yield (
            U_t[i:end], 
            U_t_plus_dt[i:end], 
            features[i:end], 
            tau[i:end], 
            f[i:end]
        )