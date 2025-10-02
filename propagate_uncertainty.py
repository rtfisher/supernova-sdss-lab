import jax
import jax.numpy as jnp
from jax import grad, vmap

# Define your mathematical operations as a function
def computation(x):
    """
    Define your mathematical operations here.
    x: input array of values
    Returns: output array y
    """
    # Example: some mathematical operations
    y = jnp.sin(x) * jnp.exp(-x**2 / 10)
    y = y + jnp.log(jnp.abs(x) + 1)
    y = y**2 / (1 + y**2)
    return y

# Function to compute output uncertainties via automatic differentiation
def propagate_uncertainty(x_values, sigma_x):
    """
    Propagate uncertainties through the computation.
    
    Args:
        x_values: array of input values
        sigma_x: array of standard deviations for inputs
    
    Returns:
        y_values: array of output values
        sigma_y: array of standard deviations for outputs
    """
    # Compute the output values
    y_values = computation(x_values)
    
    # Compute the Jacobian (derivatives dy/dx for each point)
    jacobian_fn = vmap(grad(lambda x: computation(jnp.array([x]))[0]))
    dy_dx = jacobian_fn(x_values)
    
    # Propagate uncertainties: sigma_y = |dy/dx| * sigma_x
    sigma_y = jnp.abs(dy_dx) * sigma_x
    
    return y_values, sigma_y

# Example usage
if __name__ == "__main__":
    # Input data with uncertainties
    x_i = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sigma_x_i = jnp.array([0.1, 0.15, 0.2, 0.1, 0.25])
    
    # Propagate uncertainties
    y_i, sigma_y_i = propagate_uncertainty(x_i, sigma_x_i)
    
    # Display results
    print("Input values (x_i):", x_i)
    print("Input uncertainties (σ_x_i):", sigma_x_i)
    print("\nOutput values (y_i):", y_i)
    print("Output uncertainties (σ_y_i):", sigma_y_i)
    print("\nRelative uncertainties:")
    print("  Input:", sigma_x_i / x_i * 100, "%")
    print("  Output:", sigma_y_i / jnp.abs(y_i) * 100, "%")
