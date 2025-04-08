"""
A try on learning the dissipation rate of NIOs with a NN.

using PINNs: Physic Informed Neural Networks

see example: https://github.com/Ceyron/pinns-in-jax/blob/main/poisson_pinn_in_jax_equinox.ipynb

"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt




# I used Claude for the following.
from PINNs import PhysicsInformedSolver
from dissipation_NN import DissipationNet

def train_model(data, num_epochs=100):
    """Train the oceanographic model"""
    # Extract data
    U_t = data['U_t']
    U_t_plus_dt = data['U_t_plus_dt']
    features = data['features']
    tau = data['tau']
    f = data['f']
    
    # Model dimensions
    input_channels = features.shape[1]  # Number of feature channels
    seq_length = features.shape[-1]     # Time sequence length
    hidden_dim = 64                     # Hidden dimension size
    
    # Initialize model components with random key
    key = jax.random.PRNGKey(0)
    
    # Create the neural network for dissipation
    dissipation_model = DissipationNet(
        input_channels=input_channels,
        seq_length=seq_length,
        hidden_dim=hidden_dim,
        key=key
    )
    
    # Create the physics-informed solver
    solver = PhysicsInformedSolver(
        dissipation_model=dissipation_model,
        dt=3600.0,  # 1-hour timestep
        key=key
    )
    
    # Create optimizer
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    
    # Split solver into trainable and static parts
    solver_trainable, solver_static = eqx.partition(
        solver, lambda x: eqx.is_array(x) and not (isinstance(x, jnp.ndarray) and x.size == 0)
    )
    opt_state = optimizer.init(solver_trainable)
    
    # Training loop
    losses = []
    k0_values = []
    
    # Create batch data (here we're using the full dataset as one batch for simplicity)
    # In practice, you would create multiple batches
    batch = (U_t, U_t_plus_dt, features, tau, f)
    
    # Define compiled step function
    @jax.jit
    def step(solver_trainable, solver_static, opt_state, batch):
        # Combine for computation
        solver = eqx.combine(solver_trainable, solver_static)
        
        # Define loss function
        def loss_fn(solver):
            U_t, U_t_plus_dt, features, tau, f = batch
            
            # Forward simulation
            U_pred = solver.rk4_step(U_t, features, tau, f)
            
            # Calculate physics residual
            u_mag = jnp.sqrt(U_t[:, 0]**2 + U_t[:, 1]**2)
            du_dt = (U_t_plus_dt - U_t) / solver.dt
            rhs = solver.rhs(U_t, features, tau, f, u_mag)
            residual = du_dt - rhs
            
            # Calculate losses
            data_loss = jnp.mean((U_pred - U_t_plus_dt)**2)
            physics_loss = jnp.mean(residual**2)
            k0_reg = jnp.abs(solver.K0 - 0.15)**2
            
            total_loss = data_loss + 0.1 * physics_loss + 0.01 * k0_reg
            
            return total_loss, (data_loss, physics_loss, k0_reg, U_pred)
        
        # Compute gradients
        (loss, (data_loss, physics_loss, k0_reg, U_pred)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(solver)
        
        # Extract trainable gradients
        grads_trainable, _ = eqx.partition(grads, lambda x: eqx.is_array(x) and not (isinstance(x, jnp.ndarray) and x.size == 0))
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads_trainable, opt_state)
        new_solver_trainable = eqx.apply_updates(solver_trainable, updates)
        
        # Reconstruct full solver for return
        new_solver = eqx.combine(new_solver_trainable, solver_static)
        
        return new_solver, new_opt_state, loss, data_loss, physics_loss, k0_reg, U_pred
    
    # Run training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        solver, opt_state, loss, data_loss, physics_loss, k0_reg, U_pred = step(
            solver_trainable, solver_static, opt_state, batch
        )
        
        # Re-partition for next iteration
        solver_trainable, solver_static = eqx.partition(
            solver, lambda x: eqx.is_array(x) and not (isinstance(x, jnp.ndarray) and x.size == 0)
        )
        
        # Record metrics
        losses.append(loss.item())
        k0_values.append(solver.K0.item())
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, K0 = {solver.K0.item():.6f}")
            print(f"  Data Loss: {data_loss.item():.6f}, Physics Loss: {physics_loss.item():.6f}")
    
    return solver, losses, k0_values


def evaluate_model(solver, test_data):
    """Evaluate the trained model"""
    # Extract test data
    U_t = test_data['U_t']
    U_t_plus_dt_true = test_data['U_t_plus_dt']
    features = test_data['features']
    tau = test_data['tau']
    f = test_data['f']
    
    # Forward prediction
    U_t_plus_dt_pred = solver.rk4_step(U_t, features, tau, f)
    
    # Calculate error
    mse = jnp.mean((U_t_plus_dt_pred - U_t_plus_dt_true) ** 2)
    print(f"Test MSE: {mse.item():.6f}")
    
    # Calculate the dissipation term itself
    u_mag = jnp.sqrt(U_t[:, 0]**2 + U_t[:, 1]**2)
    dissipation = solver.model(features, u_mag)
    
    # Return predictions and estimated dissipation
    return {
        'U_predicted': U_t_plus_dt_pred,
        'U_true': U_t_plus_dt_true,
        'dissipation': dissipation,
        'K0': solver.K0.item()
    }


def plot_results(training_history, evaluation_results):
    """Plot training and evaluation results"""
    losses, k0_values = training_history
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot training loss
    axs[0, 0].plot(losses)
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_yscale('log')
    
    # Plot K0 evolution
    axs[0, 1].plot(k0_values)
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('K0 Value')
    axs[0, 1].set_title('K0 Parameter Evolution')
    
    # Plot predicted vs true velocities
    U_pred = evaluation_results['U_predicted']
    U_true = evaluation_results['U_true']
    
    # Real part comparison
    axs[1, 0].scatter(U_true[:, 0], U_pred[:, 0], alpha=0.5)
    axs[1, 0].plot([-1, 1], [-1, 1], 'r--')  # Identity line
    axs[1, 0].set_xlabel('True Velocity (real)')
    axs[1, 0].set_ylabel('Predicted Velocity (real)')
    axs[1, 0].set_title('Real Component Prediction')
    axs[1, 0].axis('equal')
    
    # Imaginary part comparison
    axs[1, 1].scatter(U_true[:, 1], U_pred[:, 1], alpha=0.5)
    axs[1, 1].plot([-1, 1], [-1, 1], 'r--')  # Identity line
    axs[1, 1].set_xlabel('True Velocity (imag)')
    axs[1, 1].set_ylabel('Predicted Velocity (imag)')
    axs[1, 1].set_title('Imaginary Component Prediction')
    axs[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('ocean_model_results.png', dpi=300)
    plt.show()
    
    # Plot dissipation vs. velocity magnitude
    plt.figure(figsize=(10, 6))
    u_mag = jnp.sqrt(U_true[:, 0]**2 + U_true[:, 1]**2)
    plt.scatter(u_mag, evaluation_results['dissipation'].flatten(), alpha=0.5)
    
    # Add trendline to check for |U|³ scaling
    u_range = jnp.linspace(jnp.min(u_mag), jnp.max(u_mag), 100)
    plt.plot(u_range, 0.01 * u_range**3, 'r--', label=r'$\sim |U|^3$ Scaling')
    
    plt.xlabel('Velocity Magnitude |U| (m/s)')
    plt.ylabel('Dissipation (m/s²)')
    plt.title('Learned Dissipation vs. Velocity Magnitude')
    plt.legend()
    plt.savefig('dissipation_scaling.png', dpi=300)
    plt.show()



