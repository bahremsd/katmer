import jax.numpy as jnp # jax's numpy library we will use for all general mathematical operations
from jax import Array # Type definition for JAX arrays
from jax.typing import ArrayLike # JAX type hint for array-like objects (supports numpy, JAX arrays, etc.)

def compute_kz(nk_list: ArrayLike,   # The list of complex refractive indices (n + ik) for each medium in the multilayer thin film stack.
               angles: ArrayLike,    # The angles made by the layer normal and the light ray in each layer (in radians).
               wavelength: ArrayLike) -> Array:  # The wavelength(s) of the light incident on the multilayer thin film.

    """
    This function calculates the component of the wavevector (kz) along the z-axis (the direction normal to the surface) 
    for each layer in a multilayer thin film. It is used in the context of thin film optics, especially when applying 
    the Transfer Matrix Method (TMM) to model light propagation through multiple layers with different refractive 
    indices and incident angles. The formula computes kz for each layer based on the refractive index and the angle 
    of incidence, as well as the wavelength of the light.

    Arguments:
    - nk_list: A list or array containing the complex refractive index (n + ik) for each layer in the multilayer thin film.
      For a multilayer thin film with N layers, nk_list consists of N+2 elements. Each element is a complex value representing 
      the refractive index (n + ik) where 'n' is the real part (refractive index) and 'k' is the imaginary part (extinction coefficient).
    - angles: An array representing the angle of incidence in each layer, measured between the layer normal and the incoming light ray.
    - wavelength: A scalar value for the wavelength of the light used in the computation.

    Returns:
    - An array containing the kz component (wavevector component along the z-axis) for each layer in the thin film. 
      The kz values are computed using the formula:
      kz = (2 pi * n * cos(angle)) / wavelength, where 'n' is the complex refractive index of the layer,
      'angle' is the angle of incidence in each layer, and 'wavelength' is the wavelength of the light.
    """
    
    kz = jnp.true_divide(  # Performs element-wise division of the numerator by the denominator to calculate kz.
        jnp.multiply(  # Multiplies the complex refractive index, π, and 2.0 to form the numerator.
            jnp.multiply(jnp.array([2.0]), jnp.pi),  # Multiplies 2.0 with π (constant factor for wavevector calculation).
            jnp.multiply(nk_list, jnp.cos(angles))  # Multiplies the refractive index of each layer by the cosine of the angle.
        ), 
        wavelength  # Divides the result by the wavelength of the light to get the kz component.
    )

    return kz  # Returns the computed kz values for each layer in the multilayer thin film.

def compute_inc_layer_pass(incoherent_layer_indices: ArrayLike, layer_phases: ArrayLike) -> Array:
    """
    This function calculates the fraction of light that successfully passes through the incoherent 
    layers of an optical multilayer thin film stack. 

    The calculation is based on the exponential decay model for incoherent layers:
    Pass ratio = exp(-((4π Im[n_i cos(θ_i)]) / λ_vac))

    Also, the phase of a layer is given by:

        2π / λ * n_i * cos(θ_i) * d_i

    This term is already precomputed and given as `layer_phases` in this function. Since we have 
    already computed these phase values, we do not need to recompute them. Instead, we extract 
    the imaginary part of these phases, multiply it by -2, and apply the exponential function to 
    compute the transmission fraction:

    Pass ratio = exp(-2 * Im(layer_phases))

    Parameters:
    - incoherent_layer_indices (array-like): Indices representing which layers in the multilayer 
                                             stack are incoherent. This can be a list, numpy array, 
                                             or jax.numpy array.
    - layer_phases (array-like): Array of complex phase values for each layer in the stack. These 
                                 phases encode optical path differences and absorption effects.

    Returns:
    - Array: An array containing the computed transmission power values for the 
                         incoherent layers.
    """
    # Select values from layer_phases based on the indices of the incoherent layers
    selected_phases = jnp.take(layer_phases, incoherent_layer_indices)
    
    # Extract the imaginary part of the selected phase values (related to absorption)
    imag_phases = jnp.imag(selected_phases)
    
    # Multiply the imaginary part by -2 to match the exponential decay factor
    transformed_phases = jnp.multiply(-2, imag_phases)
    
    # Compute the exponential decay factor representing the fraction of light transmitted
    result = jnp.exp(transformed_phases)

    return result