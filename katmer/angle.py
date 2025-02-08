import jax.numpy as jnp  # Import the jax numpy module for numerical and mathematical operations, used for efficient array manipulation and computations on CPUs/GPUs/TPUs.
from jax.lax import cond  # Import the 'cond' function from jax.lax, which is used to implement conditional branching in a JAX-compatible way (similar to if-else).
from jax import Array  # Import the Array class from jax, which is used for creating arrays in JAX, though it's included here primarily for type specification purpose.
from jax.typing import ArrayLike  # Import ArrayLike from jax.typing. It is an annotation for any value that is safe to implicitly cast to a JAX array
import sys  # Import the sys module to access system-specific parameters, in this case, to retrieve the smallest representable positive number (epsilon).

# Define EPSILON as the smallest representable positive number such that 1.0 + EPSILON != 1.0
EPSILON = sys.float_info.epsilon  # Assign the machine epsilon value from the sys module

def is_forward_if_bigger_than_eps_s_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    """ 
    This function determines the propagation direction (forward or backward) for s-polarized light
    based on the refractive index and angle of incidence. It uses the complex refractive index `n` 
    (which is in the form of `n + i*k`, where `n` is the real part and `k` is the imaginary part of
    refractive index and k is representing the extinction coefficient) and the angle `theta` (in 
    radians) that the light makes with the normal to the interface.

    The function computes `n * cos(theta)` to evaluate the propagation direction for s-polarization.
    If the imaginary part of `n * cos(theta)` is positive, it indicates forward propagation (because 
    of the decay of the wave in the medium). If the imaginary part is negative, it indicates backward 
    propagation (since the wave is evanescent or decaying in the direction opposite to propagation). 

    Arguments:
    n : ArrayLike
        This is a complex jax array representing the refractive index of a material in the multilayer stack.
    
    theta : ArrayLike
        This is a jax array of angle in radians, representing the angle of incidence of the light ray 
        with respect to the normal of the layer surface. The angle is typically in the range [0, pi/2].
    
    Returns:
    Array
        This function returns a jax array of boolean value, where represents whether the propagation for 
        the corresponding angle and refractive index is forward (`True`) or backward (`False`).
    """
    
    # Calculate n * cos(theta) to evaluate propagation direction for s-polarization
    n_cos_theta = jnp.multiply(n, jnp.cos(theta))  # Multiply refractive index by cosine of angle
    
    # For evanescent or lossy mediums, forward is determined by decay
    # Determine forward propagation by checking the sign of the imaginary part of n * cos(theta)
    is_forward_s = jnp.invert(jnp.signbit(jnp.imag(n_cos_theta)))  # Invert the sign of the imaginary part
    
    # Return a boolean array where True means forward propagation, False means backward
    return is_forward_s

def is_forward_if_smaller_than_eps_s_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    """ 
    This function determines the propagation direction (forward or backward) for s-polarized light
    based on the refractive index and angle of incidence. It uses the complex refractive index `n` 
    (which is in the form of `n + i*k`, where `n` is the real part and `k` is the imaginary part of
    refractive index and k is representing the extinction coefficient) and the angle `theta` (in 
    radians) that the light makes with the normal to the interface.

    The function computes `n * cos(theta)` to evaluate the propagation direction for s-polarization.
    If the real part of `n * cos(theta)` is positive, it indicates forward propagation (because 
    of the decay of the wave in the medium). If the real part is negative, it indicates backward 
    propagation (since the wave is evanescent or decaying in the direction opposite to propagation). 

    Arguments:
    n : ArrayLike
        This is a complex jax array representing the refractive index of a material in the multilayer stack.
    
    theta : ArrayLike
        This is a jax array of angle in radians, representing the angle of incidence of the light ray 
        with respect to the normal of the layer surface. The angle is typically in the range [0, pi/2].
    
    Returns:
    Array
        This function returns a jax array of boolean value, where represents whether the propagation for 
        the corresponding angle and refractive index is forward (`True`) or backward (`False`).
    """

    # Calculate n * cos(theta) to evaluate propagation direction for s-polarization
    n_cos_theta = jnp.multiply(n, jnp.cos(theta))

    # Check if the real part of n * cos(theta) is positive or negative to determine if it's forward or backward
    is_forward_s = jnp.invert(jnp.signbit(jnp.real(n_cos_theta))) # Invert the sign of the real part

    # Return a boolean array where True means forward propagation, False means backward
    return is_forward_s

def is_forward_if_bigger_than_eps_p_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    """ 
    This function determines the propagation direction (forward or backward) for p-polarized light
    based on the refractive index and angle of incidence. It uses the complex refractive index `n` 
    (which is in the form of `n + i*k`, where `n` is the real part and `k` is the imaginary part of
    refractive index and k is representing the extinction coefficient) and the angle `theta` (in 
    radians) that the light makes with the normal to the interface.

    The function computes `n * cos(theta)` to evaluate the propagation direction for p-polarization.
    If the imaginary part of `n * cos(theta)` is positive, it indicates forward propagation (because 
    of the decay of the wave in the medium). If the imaginary part is negative, it indicates backward 
    propagation (since the wave is evanescent or decaying in the direction opposite to propagation). 

    Arguments:
    n : ArrayLike
        This is a complex jax array representing the refractive index of a material in the multilayer stack.
    
    theta : ArrayLike
        This is a jax array of angle in radians, representing the angle of incidence of the light ray 
        with respect to the normal of the layer surface. The angle is typically in the range [0, pi/2].
    
    Returns:
    Array
        This function returns a jax array of boolean value, where represents whether the propagation for 
        the corresponding angle and refractive index is forward (`True`) or backward (`False`).
    """

    # Calculate n * cos(theta) to evaluate propagation direction for s-polarization
    n_cos_theta = jnp.multiply(n, jnp.cos(theta)) # Multiply refractive index by cosine of angle

    # For evanescent or lossy mediums, forward is determined by decay
    # Determine forward propagation by checking the sign of the imaginary part of n * cos(theta)
    is_forward_p = jnp.invert(jnp.signbit(jnp.imag(n_cos_theta))) # Invert the sign of the imaginary part

    # Return a boolean array where True means forward propagation, False means backward
    return is_forward_p

def is_forward_if_smaller_than_eps_p_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    """ 
    This function determines the propagation direction (forward or backward) for p-polarized light
    based on the refractive index and angle of incidence. It uses the complex refractive index `n` 
    (which is in the form of `n + i*k`, where `n` is the real part and `k` is the imaginary part of
    refractive index and k is representing the extinction coefficient) and the angle `theta` (in 
    radians) that the light makes with the normal to the interface.

    The function computes `n * cos(theta)` to evaluate the propagation direction for p-polarization.
    If the real part of `n * cos(theta)` is positive, it indicates forward propagation (because 
    of the decay of the wave in the medium). If the real part is negative, it indicates backward 
    propagation (since the wave is evanescent or decaying in the direction opposite to propagation). 

    Arguments:
    n : ArrayLike
        This is a complex jax array representing the refractive index of a material in the multilayer stack.
    
    theta : ArrayLike
        This is a jax array of angle in radians, representing the angle of incidence of the light ray 
        with respect to the normal of the layer surface. The angle is typically in the range [0, pi/2].
    
    Returns:
    Array
        This function returns a jax array of boolean value, where represents whether the propagation for 
        the corresponding angle and refractive index is forward (`True`) or backward (`False`).
    """

    # Calculate n * cos(theta) to evaluate propagation direction for p-polarization
    n_cos_theta_star = jnp.multiply(n, jnp.cos(jnp.conj(theta)))

    # Check if the real part of n * cos(theta) is positive or negative to determine if it's forward or backward
    is_forward_p = jnp.invert(jnp.signbit(jnp.real(n_cos_theta_star))) # Invert the sign of the real part

    # Return a boolean array where True means forward propagation, False means backward
    return is_forward_p

def is_propagating_wave_s_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    """
    This function determines whether light propagating in a layer of a multilayer thin film structure 
    is traveling forward or backward for s-polarized light. The decision is made based on the imaginary 
    part of the product of the refractive index (`n`) and the cosine of the angle of incidence (`theta`). 
    If the absolute value of the imaginary component exceeds a threshold (1000 * EPSILON), 
    the function applies one of two conditions to decide the propagation direction. 
    
    Arguments:
    ----------
    n: ArrayLike
        The refractive index (complex) of the medium in the layer under consideration. This could have 
        a real part (indicating normal propagation) and/or an imaginary part (indicating losses or evanescent waves).
    
    theta: ArrayLike
        The angle of incidence (in radians) of the light wave within the layer. This is used to compute 
        the cosine term for further evaluation.
    
    Returns:
    --------
    Array
        A boolean value or array of boolean values (`True` or `False`) indicating whether the light wave 
        is propagating forward (`True`) or backward (`False`) for s-polarization.
    
    Functionality:
    --------------
    The function first evaluates whether the product of `n * cos(theta)` has a sufficiently large imaginary part.
    Based on this condition, it invokes one of two helper functions (`is_forward_if_bigger_than_eps_s_pol` or 
    `is_forward_if_smaller_than_eps_s_pol`) using `jax.lax.cond` for efficient branching. The output is determined 
    by the specific conditions defined in these helper functions.
    """

    # Calculate the condition by checking the absolute value of the imaginary part of `n * cos(theta)`.
    # If the absolute value exceeds `1000 * EPSILON`, it is assumed that the wave is evanescent or experiencing loss.
    condition = jnp.squeeze(jnp.greater(jnp.abs(jnp.imag(jnp.multiply(n, jnp.cos(theta)))), jnp.multiply(jnp.array([EPSILON]), jnp.array([1e3]))))
    
    # Use the `jax.lax.cond` function to efficiently choose between two helper functions.
    # If the condition is True, call `is_forward_if_bigger_than_eps_s_pol` with `n` and `theta`.
    # Otherwise, call `is_forward_if_smaller_than_eps_s_pol` with `n` and `theta`.
    is_forward_s = cond(condition, is_forward_if_bigger_than_eps_s_pol, is_forward_if_smaller_than_eps_s_pol, n, theta)
    
    # Return the boolean value indicating whether the wave is forward-propagating for s-polarization.
    return is_forward_s

def is_propagating_wave_p_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    """
    This function determines whether light propagating in a layer of a multilayer thin film structure 
    is traveling forward or backward for p-polarized light. The decision is made based on the imaginary 
    part of the product of the refractive index (`n`) and the cosine of the angle of incidence (`theta`). 
    If the absolute value of the imaginary component exceeds a threshold (1000 * EPSILON), 
    the function applies one of two conditions to decide the propagation direction. 
    
    Arguments:
    ----------
    n: ArrayLike
        The refractive index (complex) of the medium in the layer under consideration. This could have 
        a real part (indicating normal propagation) and/or an imaginary part (indicating losses or evanescent waves).
    
    theta: ArrayLike
        The angle of incidence (in radians) of the light wave within the layer. This is used to compute 
        the cosine term for further evaluation.
    
    Returns:
    --------
    Array
        A boolean value or array of boolean values (`True` or `False`) indicating whether the light wave 
        is propagating forward (`True`) or backward (`False`) for p-polarization.
    
    Functionality:
    --------------
    The function first evaluates whether the product of `n * cos(theta)` has a sufficiently large imaginary part.
    Based on this condition, it invokes one of two helper functions (`is_forward_if_bigger_than_eps_p_pol` or 
    `is_forward_if_smaller_than_eps_p_pol`) using `jax.lax.cond` for efficient branching. The output is determined 
    by the specific conditions defined in these helper functions.
    """

    # Calculate the condition by checking the absolute value of the imaginary part of `n * cos(theta)`.
    # If the absolute value exceeds `1000 * EPSILON`, it is assumed that the wave is evanescent or experiencing loss.
    condition = jnp.squeeze(jnp.greater(jnp.abs(jnp.imag(jnp.multiply(n, jnp.cos(theta)))), jnp.multiply(jnp.array([EPSILON]), jnp.array([1e3]))))
    
    # Use the `jax.lax.cond` function to efficiently choose between two helper functions.
    # If the condition is True, call `is_forward_if_bigger_than_eps_p_pol` with `n` and `theta`.
    # Otherwise, call `is_forward_if_smaller_than_eps_p_pol` with `n` and `theta`.
    is_forward_p = cond(condition, is_forward_if_bigger_than_eps_p_pol, is_forward_if_smaller_than_eps_p_pol, n, theta)
    
    # Return the boolean value indicating whether the wave is forward-propagating for p-polarization.
    return is_forward_p

def update_theta_arr_incoming(theta_array: ArrayLike) -> Array:
    """
    This function adjusts the angle of incidence for a multilayer thin-film system by ensuring that 
    the light in the incoming medium propagates in a forward direction. It operates on a vector 
    (`theta_array`) containing the angles that light makes with the normal in each layer of the multilayer 
    structure. The theta_array values depend on the incident angle and the wavelength-dependent refractive 
    index (`n(wavelength)`).

    Specifically, if the wave in the incoming medium (the first layer) is not forward propagating, 
    the function modifies the first element of the theta_array vector to its complement with respect to 180 degrees 
    (pi in radians). This ensures that the light is in an optically correct format, aligning with the 
    physical constraints of multilayer systems where light propagation must conform to specific conditions 
    such as forward propagation in the infinitely thick incoming medium.

    Arguments:
    ----------
    theta_array : ArrayLike
        A vector containing the angles (in radians) that light makes with the normal in each layer 
        of a multilayer thin-film structure. The first element corresponds to the incoming medium.

    Returns:
    --------
    Array
        The updated `theta_array` with the first angle corrected (if necessary) to ensure forward 
        propagation in the incoming medium.
    """

    # Access the first element of theta_array and subtract it from pi (180 degrees in radians).
    # This operation ensures the first angle is adjusted to its complement with respect to pi
    # if it is not forward propagating.
    return theta_array.at[0].set(jnp.pi - theta_array.at[0].get())

def update_theta_arr_outgoing(theta_array: ArrayLike) -> Array:
    """
    This function updates the last element of the angle vector (`theta_array`) to ensure
    the outgoing wave in a multilayer thin-film structure adheres to optical conventions.

    In multilayer thin films, the angles in `theta_array` represent the angles of light with respect
    to the normal in each layer. These angles are influenced by the angle of incidence and the
    wavelength because the refractive index is wavelength-dependent (n(wavelength)).

    The outgoing medium is assumed to have a forward-propagating wave because multilayer thin films
    with complex refractive indices (n + ik) typically don't support negative refractive indices.
    To ensure this optical consistency, the last angle in the `theta_array` (which represents the
    outgoing medium) is updated to its complementary angle. Specifically, the last element is set 
    to `pi - theta_array[-1]`, which effectively adjusts the angle for optical correctness.

    This adjustment is critical for ensuring the outgoing light does not deviate excessively 
    from the normal after passing through a multilayer structure, maintaining a physically and 
    optically consistent representation of the outgoing wave.

    Args:
        theta_array (ArrayLike): A vector containing the angles of light with respect to the 
        normal in each layer of the multilayer thin-film structure.

    Returns:
        Array: The updated angle vector (`theta_array`) with its last element adjusted to ensure
        optical consistency for the outgoing wave.
    """

    # Access the last element of theta_array and subtract it from pi (180 degrees in radians).
    # This operation ensures the last angle is adjusted to its complement with respect to pi
    # if it is not forward propagating.
    return theta_array.at[-1].set(jnp.pi - theta_array.at[-1].get())

def return_unchanged_theta(theta_array: ArrayLike) -> Array:
    """
    This function is a placeholder or pseudo-function designed to return the input array `theta_array` without any modifications. 
    It is primarily used when working with JAX's `lax.cond` function, where different branches (functions) need to be specified 
    for conditional execution. 

    In contexts where we do not want to alter the `theta_array` during conditional branching, this function serves as an 
    identity function. For example, if light forward propagates in certain scenarios where the outgoing and incoming functions 
    of `theta_array` do not require modification, this function ensures `theta_array` remains unchanged.

    Arguments:
    -----------
    theta_array: ArrayLike
        The input array representing `theta`, which might correspond to angular values or other parameter arrays in a 
        computation. The function treats this array as immutable within its context.

    Returns:
    --------
    Array
        The same input array, `theta_array`, returned unchanged.
    """
    return theta_array

def compute_layer_angles_s_pol(angle_of_incidence: ArrayLike,
                               nk_list: ArrayLike) -> Array:
    """
    This function calculates the angles that light makes with the layer normal in each layer of a multilayer stack 
    structure under s-polarized light. The function uses Snell's law to compute the angles for each layer, ensuring 
    compatibility with evanescent and lossy cases by handling complex refractive indices. It validates the forward 
    propagation conditions for the incoming and outgoing media and updates the corresponding angles in the theta array
    if necessary. The resulting array helps determine the behavior of light within the multilayer system. 

    Arguments:
    - `angle_of_incidence` (ArrayLike): This is the angle at which light is incident on the first layer of the multilayer 
      stack. It is expected to be in radians.
    - `nk_list` (ArrayLike): A JAX array containing the complex refractive indices of the multilayer system, including 
      the incoming medium, each layer in the stack, and the outgoing medium. 
      - For a multilayer system with N layers, this array has a length of N+2: 
        [n_incoming, n_layer_0, n_layer_1, ..., n_layer_N, n_outgoing].
      - The complex refractive indices account for both the real part (refraction) and the imaginary part 
        (absorption/losses).

    Returns:
    - A JAX array (`theta_array`) of the same length as `nk_list`, where each element represents the calculated angle 
      (in radians) for each layer of the multilayer structure, including the incoming and outgoing media.
    """

    # Calculate the sine of the angles by using the first layer n * sin(theta)
    # sin(theta_i) = (n_0 * sin(theta_in)) / n_i, where n_0 is the refractive index of the first layer
    sin_theta = jnp.true_divide(jnp.multiply(jnp.sin(angle_of_incidence), nk_list.at[0].get()), nk_list)
    
    # Compute the angle (theta) in each layer using the arcsin function
    # jnp.arcsin is used here as it supports complex values if necessary, allowing for accurate calculations 
    # in lossy or evanescent cases
    theta_array = jnp.arcsin(sin_theta)

    # Check if the wave in the incoming medium is propagating
    # This determines whether the computed angle is forward-facing or not
    incoming_props = is_propagating_wave_s_pol(nk_list.at[0].get(), theta_array.at[0].get())

    # Check if the wave in the outgoing medium is propagating
    # This validates the forward propagation condition for the outgoing medium
    outgoing_props = is_propagating_wave_s_pol(nk_list.at[-1].get(), theta_array.at[-1].get())

    # Create conditions to check if the incoming wave is propagating
    # The conditions ensure that the calculated angles are correctly oriented
    condition_incoming = jnp.array_equal(incoming_props, jnp.array([True], dtype=bool))
    condition_outgoing = jnp.array_equal(outgoing_props, jnp.array([True], dtype=bool))

    # Update the angle for the incoming medium using JAX's conditional function (jax.lax.cond)
    # If the condition is true, the theta array is modified for the incoming medium
    theta_array = cond(condition_incoming, update_theta_arr_incoming, return_unchanged_theta, operand=theta_array)
    
    # Update the angle for the outgoing medium using JAX's conditional function (jax.lax.cond)
    # If the condition is true, the theta array is modified for the outgoing medium
    theta_array = cond(condition_outgoing, update_theta_arr_outgoing, return_unchanged_theta, operand=theta_array)
    #print("third theta : ", jnp.asarray(theta_array))

    # Return a 1D theta array for each layer
    # This array contains the calculated angles for all layers
    return theta_array

def compute_layer_angles_p_pol(angle_of_incidence: ArrayLike,
                               nk_list: ArrayLike) -> Array:
    """
    This function calculates the angles that light makes with the layer normal in each layer of a multilayer stack 
    structure under p-polarized light. The function uses Snell's law to compute the angles for each layer, ensuring 
    compatibility with evanescent and lossy cases by handling complex refractive indices. It validates the forward 
    propagation conditions for the incoming and outgoing media and updates the corresponding angles in the theta array
    if necessary. The resulting array helps determine the behavior of light within the multilayer system. 

    Arguments:
    - `angle_of_incidence` (ArrayLike): This is the angle at which light is incident on the first layer of the multilayer 
      stack. It is expected to be in radians.
    - `nk_list` (ArrayLike): A JAX array containing the complex refractive indices of the multilayer system, including 
      the incoming medium, each layer in the stack, and the outgoing medium. 
      - For a multilayer system with N layers, this array has a length of N+2: 
        [n_incoming, n_layer_0, n_layer_1, ..., n_layer_N, n_outgoing].
      - The complex refractive indices account for both the real part (refraction) and the imaginary part 
        (absorption/losses).

    Returns:
    - A JAX array (`theta_array`) of the same length as `nk_list`, where each element represents the calculated angle 
      (in radians) for each layer of the multilayer structure, including the incoming and outgoing media.
    """

    # Calculate the sine of the angles by using the first layer n * sin(theta)
    # sin(theta_i) = (n_0 * sin(theta_in)) / n_i, where n_0 is the refractive index of the first layer
    sin_theta = jnp.true_divide(jnp.multiply(jnp.sin(angle_of_incidence), nk_list.at[0].get()), nk_list)
    
    # Compute the angle (theta) in each layer using the arcsin function
    # jnp.arcsin is used here as it supports complex values if necessary, allowing for accurate calculations 
    # in lossy or evanescent cases
    theta_array = jnp.arcsin(sin_theta)

    # Check if the wave in the incoming medium is propagating
    # This determines whether the computed angle is forward-facing or not
    incoming_props = is_propagating_wave_p_pol(nk_list.at[0].get(), theta_array.at[0].get())

    # Check if the wave in the outgoing medium is propagating
    # This validates the forward propagation condition for the outgoing medium
    outgoing_props = is_propagating_wave_p_pol(nk_list.at[-1].get(), theta_array.at[-1].get())

    # Create conditions to check if the incoming wave is propagating
    # The conditions ensure that the calculated angles are correctly oriented
    condition_incoming = jnp.array_equal(incoming_props, jnp.array([True], dtype=bool))
    condition_outgoing = jnp.array_equal(outgoing_props, jnp.array([True], dtype=bool))

    # Update the angle for the incoming medium using JAX's conditional function (jax.lax.cond)
    # If the condition is true, the theta array is modified for the incoming medium
    theta_array = cond(condition_incoming, update_theta_arr_incoming, return_unchanged_theta, operand=theta_array)

    # Update the angle for the outgoing medium using JAX's conditional function (jax.lax.cond)
    # If the condition is true, the theta array is modified for the outgoing medium
    theta_array = cond(condition_outgoing, update_theta_arr_outgoing, return_unchanged_theta, operand=theta_array)

    # Return a 1D theta array for each layer
    # This array contains the calculated angles for all layers
    return theta_array

def compute_layer_angles(angle_of_incidence: ArrayLike,
                         nk_list: ArrayLike,
                         polarization: ArrayLike) -> Array:
    """
    This function calculates the angles of wave propagation within a multilayer system 
    for a given angle of incidence and polarization state. It selects the appropriate 
    function to compute the layer angles based on the type of polarization (s-polarization or p-polarization).
    The function uses `jnp.select` to dynamically choose the appropriate subfunction
    (`compute_layer_angles_s_pol` or `compute_layer_angles_p_pol`) based on the value of the `polarization` array. 
    These subfunctions are responsible for computing the angles for s-polarization or p-polarization, respectively.

    Arguments:
    - angle_of_incidence (ArrayLike): The angle of incidence of the incoming wave, given as a scalar value. 
      This angle is typically measured relative to the normal of the first interface in the multilayer structure.
    - nk_list (ArrayLike): A JAX array containing the complex refractive indices for all layers in the 
      multilayer system. The array structure is as follows:
        [n_incoming, n_layer_0, n_layer_1, ..., n_layer_N, n_outgoing], where:
        - `n_incoming` is the refractive index of the incoming medium,
        - `n_layer_0, n_layer_1, ..., n_layer_N` are the refractive indices of the individual layers, and
        - `n_outgoing` is the refractive index of the outgoing medium.
      The refractive indices can be complex, where the real part represents the refractive index 
      and the imaginary part accounts for material absorption (extinction coefficient).
    - polarization (ArrayLike): A binary JAX array indicating the type of polarization:
        - If the array is [False], it represents s-polarization, where the electric field 
          is perpendicular to the plane of incidence.
        - If the array is [True], it represents p-polarization, where the electric field 
          is parallel to the plane of incidence.

    Returns:
    - Array: A JAX array containing the calculated angles of wave propagation within each layer of the multilayer 
      structure. The specific angles depend on the type of polarization and the refractive indices of the layers.
    """
    
    return jnp.select(
        condlist=[
            jnp.array_equal(polarization, jnp.array([False], dtype=bool)),  # Check for s-polarization
            jnp.array_equal(polarization, jnp.array([True], dtype=bool))   # Check for p-polarization
        ],
        choicelist=[
            compute_layer_angles_s_pol(angle_of_incidence, nk_list),  # Call the s-polarization function
            compute_layer_angles_p_pol(angle_of_incidence, nk_list)   # Call the p-polarization function
        ]
    )