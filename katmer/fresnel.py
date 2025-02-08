import jax.numpy as jnp  # Import the jax numpy module for numerical and mathematical operations, used for efficient array manipulation and computations on CPUs/GPUs/TPUs.
from jax import Array  # Import the Array class from jax, which is used for creating arrays in JAX, though it's included here primarily for type specification purpose.
from jax.typing import ArrayLike  # Import ArrayLike from jax.typing. It is an annotation for any value that is safe to implicitly cast to a JAX array

def fresnel_s(first_layer_n: ArrayLike,
              second_layer_n: ArrayLike,
              first_layer_theta: ArrayLike,
              second_layer_theta: ArrayLike) -> Array:
    """
    This function calculates the Fresnel reflection (r_s) and transmission (t_s) coefficients
    for s-polarized light (electric field perpendicular to the plane of incidence) at the interface
    between two materials. The inputs are the refractive indices and the angles of incidence and
    refraction for the two layers.

    Args:
        first_layer_n (ArrayLike): Refractive index of the first layer (incident medium).
            This can be a single value or an array if computing for multiple incident angles/materials.
        second_layer_n (ArrayLike): Refractive index of the second layer (transmitted medium).
            Similar to the first argument, this can be a single value or an array.
        first_layer_theta (ArrayLike): Angle of incidence in the first layer (in radians).
            This can be a single value or an array.
        second_layer_theta (ArrayLike): Angle of refraction in the second layer (in radians).
            This can be a single value or an array.

    Returns:
        Array: A JAX array containing two elements:
            - r_s: The Fresnel reflection coefficient for s-polarized light.
            - t_s: The Fresnel transmission coefficient for s-polarized light.

    Function Explanation:
    This function implements the Fresnel equations specifically for s-polarized light, where the electric
    field is perpendicular to the plane of incidence. These equations describe how light interacts at the
    boundary between two different optical media. The reflection coefficient, `r_s`, measures the ratio
    of reflected to incident electric field amplitude, while the transmission coefficient, `t_s`, measures
    the ratio of transmitted to incident electric field amplitude. These coefficients depend on the refractive
    indices of the two media and the angles of incidence and refraction. The function returns a single JAX 
    array containing both coefficients, enabling compatibility with differentiable programming and GPU/TPU 
    acceleration.
    """

    cos_first_theta = jnp.cos(first_layer_theta)  # Calculate the cosine of the incident angle (first layer).
    cos_second_theta = jnp.cos(second_layer_theta)  # Calculate the cosine of the refraction angle (second layer).
    
    # Multiply the refractive index of the first layer by the cosine of its angle.
    first_ncostheta = jnp.multiply(first_layer_n, cos_first_theta)
    
    # Multiply the refractive index of the second layer by the cosine of its angle.
    second_ncostheta = jnp.multiply(second_layer_n, cos_second_theta)
    
    # Compute the sum of the two terms: n1*cos(theta1) + n2*cos(theta2).
    add_ncosthetas = jnp.add(first_ncostheta, second_ncostheta)

    # Calculate the reflection coefficient (r_s) for s-polarized light.
    # Formula: r_s = (n1*cos(theta1) - n2*cos(theta2)) / (n1*cos(theta1) + n2*cos(theta2))
    # This represents the fraction of light that is reflected at the interface.
    r_s = jnp.true_divide(jnp.subtract(first_ncostheta, second_ncostheta), add_ncosthetas)

    # Calculate the transmission coefficient (t_s) for s-polarized light.
    # Formula: t_s = 2*n1*cos(theta1) / (n1*cos(theta1) + n2*cos(theta2))
    # This represents the fraction of light that passes through the interface.
    t_s = jnp.true_divide(jnp.multiply(2, first_ncostheta), add_ncosthetas)

    # Combine the reflection and transmission coefficients into a single array.
    # The resulting array contains both coefficients for easy use in further computations.
    return jnp.stack([r_s, t_s])  # Return a stacked array containing r_s and t_s coefficients.

def fresnel_p(first_layer_n: ArrayLike,
              second_layer_n: ArrayLike,
              first_layer_theta: ArrayLike,
              second_layer_theta: ArrayLike) -> Array:
    """
    This function calculates the Fresnel reflection (r_p) and transmission (t_p) coefficients
    for p-polarized light at the interface between two different media. It uses the refractive indices
    of the two media (_first_layer_n and _second_layer_n) and the incident and transmitted angles
    (_first_layer_theta and _second_layer_theta) to compute these values.

    Args:
        _first_layer_n: ArrayLike Refractive index of the first medium (can be float or ndarray).
        _second_layer_n: ArrayLike Refractive index of the second medium (can be float or ndarray).
        _first_layer_theta: ArrayLike Incident angle (in radians) in the first medium (can be float or ndarray).
        _second_layer_theta: ArrayLike Transmitted angle (in radians) in the second medium (can be float or ndarray).

    Returns:
        Array: A tuple containing two arrays:
            - r_p: The reflection coefficient for p-polarized light.
            - t_p: The transmission coefficient for p-polarized light.
    """
    cos_first_theta = jnp.cos(first_layer_theta)  # Calculate the cosine of the incident angle in the first medium
    cos_second_theta = jnp.cos(second_layer_theta)  # Calculate the cosine of the transmitted angle in the second medium

    second_n_first_costheta = jnp.multiply(second_layer_n, cos_first_theta)  # Multiply the refractive index of the second medium with the cosine of the incident angle
    first_n_second_costheta = jnp.multiply(first_layer_n, cos_second_theta)  # Multiply the refractive index of the first medium with the cosine of the transmitted angle

    add_ncosthetas = jnp.add(second_n_first_costheta, first_n_second_costheta)  # Add the two previous results together

    # Calculate the reflection coefficient for p-polarized light (r_p)
    # This equation is based on the Fresnel equations for p-polarization, where
    # r_p is the ratio of the reflected and incident electric field amplitudes for p-polarized light.
    r_p = jnp.true_divide(jnp.subtract(second_n_first_costheta, first_n_second_costheta), add_ncosthetas)
    # Subtract the second result from the first result, divide by the sum calculated earlier to get r_p.

    # Calculate the transmission coefficient for p-polarized light (t_p)
    # This equation is also derived from the Fresnel equations for p-polarization.
    # t_p represents the ratio of the transmitted and incident electric field amplitudes.
    t_p = jnp.true_divide(jnp.multiply(2, jnp.multiply(first_layer_n, cos_first_theta)), add_ncosthetas)
    # Multiply the refractive index of the first medium with the cosine of the incident angle, then multiply by 2.
    # Divide the result by the sum calculated earlier to get t_p.

    # Return the reflection and transmission coefficients as a tuple of jnp arrays
    # Both r_p and t_p are essential for understanding how light interacts with different layers.
    return jnp.stack([r_p, t_p])  # Stack r_p and t_p into a single array (tuple) and return them