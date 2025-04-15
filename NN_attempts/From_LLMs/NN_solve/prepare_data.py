import jax
import jax.numpy as jnp

def prepare_oceanographic_data():
    """
    Prepare oceanographic data for model training
    Returns time series of currents, wind stress, and features
    """
    # In a real scenario, you would load data from files/databases
    # This is a simplified example with synthetic data
    
    # Create a key for reproducible random data
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Sample parameters
    num_samples = 100 #1000
    seq_length = 24  # 24-hour sequences
    grid_size = (20, 20)  # 20x20 spatial grid
    
    # Generate synthetic oceanographic data
    
    # 1. Complex current velocities (U) at time t and t+dt
    # U shape: [num_samples, 2], where [:, 0] is real part, [:, 1] is imaginary part
    U_t = jax.random.normal(key1, (num_samples, 2)) * 0.5  # m/s velocities
    
    # Simulate next timestep with some realistic evolution
    # In reality, this would be your observed data from measurements
    U_t_plus_dt = U_t + jax.random.normal(key2, (num_samples, 2)) * 0.1
    
    # 2. Coriolis parameter (varies with latitude)
    # f = 2Ω*sin(latitude), where Ω is Earth's rotation rate
    # Value around 1e-4 s^-1 in mid-latitudes
    f = jnp.ones(num_samples) * 1e-4  
    
    # 3. Wind stress
    # tau shape: [num_samples, 2], similar to U
    tau = jax.random.normal(key3, (num_samples, 2)) * 0.1  # N/m²
    
    # 4. Features for neural network
    # Shape: [num_samples, channels, height, width, time]
    # These would include current velocities, gradients, etc.
    num_features = 5  # e.g., U, dU/dx, dU/dy, wind, stratification
    
    # Create simple synthetic features
    features = jnp.zeros((num_samples, num_features, *grid_size, seq_length))
    
    # Add some patterns to features (this is highly simplified)
    for i in range(num_samples):
        for j in range(num_features):
            # Create spatiotemporal patterns
            for t in range(seq_length):
                features = features.at[i, j, :, :, t].set(
                    jnp.sin(jnp.linspace(0, 3, grid_size[0]))[:, None] * 
                    jnp.cos(jnp.linspace(0, 3, grid_size[1]))[None, :] * 
                    (t/seq_length)
                )
    
    return {
        'U_t': U_t,
        'U_t_plus_dt': U_t_plus_dt,
        'features': features,
        'tau': tau,
        'f': f
    }