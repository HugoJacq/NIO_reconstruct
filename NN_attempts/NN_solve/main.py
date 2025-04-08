import jax

from prepare_data import prepare_oceanographic_data
from training import train_model, evaluate_model, plot_results

"""Main function to run the entire pipeline"""
# Set JAX precision and memory settings
jax.config.update('jax_enable_x64', True)  # Use double precision

# Prepare data
print("Preparing data...")
data = prepare_oceanographic_data()

# Split into train/test (simple 80/20 split for this example)
n = data['U_t'].shape[0]
train_idx = int(0.8 * n)

train_data = {k: v[:train_idx] for k, v in data.items()}
test_data = {k: v[train_idx:] for k, v in data.items()}

# Train model
print("\nTraining model...")
solver, losses, k0_values = train_model(train_data, num_epochs=200)

# Evaluate on test data
print("\nEvaluating model...")
evaluation_results = evaluate_model(solver, test_data)

# Plot results
plot_results((losses, k0_values), evaluation_results)

# Print final K0 value
print(f"\nFinal estimated K0: {evaluation_results['K0']:.6f}")