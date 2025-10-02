import jax                                                                                    
import jax.numpy as jnp
from jax import grad, vmap

# Standard candle parameters (soft parameters)
M_V = -19.0  # Absolute magnitude of Type Ia SN
sigma_M_V = 1.0  # Uncertainty in absolute magnitude

# Define your mathematical operations as a function
def computation(m_apparent, M_abs):
    """
    Calculate distance modulus and convert to distance in parsecs.
    
    Distance modulus: μ = m - M = 5 * log10(d/pc) - 5
    Solving for distance: d = 10^((m - M + 5) / 5) parsecs
    
    Args:
        m_apparent: apparent magnitude
        M_abs: absolute magnitude
    
    Returns:
        distance in Megaparsecs (Mpc)
    """
    distance_modulus = m_apparent - M_abs
    distance_pc = 10**((distance_modulus + 5) / 5)
    distance_Mpc = distance_pc / 1e6  # Convert parsecs to Megaparsecs
    return distance_Mpc

# Function to compute output uncertainties via automatic differentiation
def propagate_uncertainty(m_apparent, sigma_m, M_abs, sigma_M):
    """
    Propagate uncertainties through the distance calculation.
    Accounts for uncertainties in both apparent magnitude and absolute magnitude.
    
    Args:
        m_apparent: array of apparent magnitudes
        sigma_m: array of uncertainties in apparent magnitudes
        M_abs: absolute magnitude (scalar)
        sigma_M: uncertainty in absolute magnitude (scalar)
    
    Returns:
        distances: array of distances in Mpc
        sigma_distances: array of uncertainties in distances (Mpc)
    """
    # Compute the distances
    distances = computation(m_apparent, M_abs)
    
    # Compute derivatives with respect to apparent magnitude
    grad_m_fn = vmap(lambda m: grad(lambda x: computation(x, M_abs))(m))
    d_dist_dm = grad_m_fn(m_apparent)
    
    # Compute derivative with respect to absolute magnitude
    grad_M_fn = vmap(lambda m: grad(lambda M: computation(m, M))(M_abs))
    d_dist_dM = grad_M_fn(m_apparent)
    
    # Propagate uncertainties (assuming independent errors):
    # sigma_d^2 = (∂d/∂m)^2 * sigma_m^2 + (∂d/∂M)^2 * sigma_M^2
    sigma_distances = jnp.sqrt(
        (d_dist_dm * sigma_m)**2 + 
        (d_dist_dM * sigma_M)**2
    )
    
    return distances, sigma_distances

# Example usage
if __name__ == "__main__":
    # Input data: apparent magnitudes with uncertainties
    m_apparent_i = jnp.array([15.0, 16.5, 18.0, 19.5, 21.0])
    sigma_m_i = jnp.array([0.1, 0.15, 0.2, 0.25, 0.3])
    
    # Propagate uncertainties
    distances_i, sigma_distances_i = propagate_uncertainty(
        m_apparent_i, sigma_m_i, M_V, sigma_M_V
    )
    
    # Display results
    print("=" * 60)
    print("Type Ia Supernova Distance Calculation")
    print("=" * 60)                                                                           
    print(f"\nStandard candle: M_V = {M_V} ± {sigma_M_V}")
    print("\nInput apparent magnitudes (m):", m_apparent_i)
    print("Input uncertainties (σ_m):", sigma_m_i)
    print("\nCalculated distances (Mpc):", distances_i)
    print("Distance uncertainties (σ_d, Mpc):", sigma_distances_i)
    print("\nRelative uncertainties (%):")
    for i in range(len(distances_i)):
        rel_unc = (sigma_distances_i[i] / distances_i[i]) * 100
        print(f"  SN #{i+1}: {rel_unc:.2f}%")    
