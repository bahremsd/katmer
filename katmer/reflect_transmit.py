import jax.numpy as jnp  # jax's numpy library we will use for all general mathematical operations
from jax import vmap  # We will vectorize while obtaining rt, that's why we import it
from jax import Array  # Type definition for JAX arrays
from jax.typing import ArrayLike  # JAX type hint for array-like objects (supports numpy, JAX arrays, etc.)

from .fresnel import fresnel_s, fresnel_p  # We will use our custom fresnel_s, fresnel_p functions while calculating r, t
from .cascaded_matmul import coh_cascaded_matrix_multiplication  # We will use this function while calculating coh layers in incoh film

def calculate_reflectance_from_coeff(r: ArrayLike) -> Array:
    """ 
    This function calculates the reflectance given the reflection coefficient 'r'.
    
    Arguments:
    r : ArrayLike
        - This argument represents the reflection coefficient, which is typically a complex number that describes how much light is 
          reflected when it encounters a surface or interface. 
        - It can be a scalar or an array of reflection coefficients for multiple values.
        - The input can be a list, tuple, or any other object that can be converted to a jax array, hence the use of 'ArrayLike'.
    
    Returns:
    Array
        - The function returns the reflectance, which is calculated as the square of the absolute value of 'r'.
        - Reflectance is a measure of the fraction of light that is reflected by a surface, and is a real, non-negative quantity.
    """
    return jnp.square(jnp.abs(r))  # Calculates the square of the absolute value of 'r', i.e., reflectance.

def calculate_transmittace_from_coeff_s_pol(t: ArrayLike,
                                            nk_first_layer_of_slab: ArrayLike,
                                            angle_first_layer_of_slab: ArrayLike,
                                            nk_last_layer_of_slab: ArrayLike,
                                            angle_last_layer_of_slab: ArrayLike) -> Array:
    """
    This function calculates the transmittance (T) of light with s-polarization through a slab based on the given 
    transmission coefficient `t`, refractive indices of the first and last layers of the slab, and their corresponding 
    angles of incidence or transmission. Physically, this ratio accounts for the matching of impedance 
    between the layers and the energy conservation in the system. The result is a measure of how much light energy 
    successfully passes through the slab relative to the initial incident energy.

    Arguments:
    t: ArrayLike
        The complex transmission coefficient for s-polarized light. This represents the ratio of the transmitted 
        electric field to the incident electric field.
    nk_first_layer_of_slab: ArrayLike
        The complex refractive index of the first layer of the slab. The refractive index (n + ik) combines the real 
        part (n) that determines the speed of light in the material and the imaginary part (k) that accounts for 
        absorption losses.
    angle_first_layer_of_slab: ArrayLike
        The angle of incidence of light in the first layer of the slab, expressed in radians.
    nk_last_layer_of_slab: ArrayLike
        The complex refractive index of the last layer of the slab, which governs the light's behavior as it exits 
        the slab.
    angle_last_layer_of_slab: ArrayLike
        The angle of refraction or transmission in the last layer of the slab, expressed in radians.

    Returns:
    T: Array
        The calculated transmittance (T) for the given slab. Transmittance is the proportion of the incident light 
        intensity that passes through the slab, and it is calculated using the transmission coefficient and the 
        refractive indices of the involved layers.
    """
    T = jnp.multiply(jnp.square(jnp.abs(t)), jnp.true_divide(jnp.real(jnp.multiply(nk_last_layer_of_slab, jnp.cos(angle_last_layer_of_slab))),
                                                             jnp.real(jnp.multiply(nk_first_layer_of_slab, jnp.cos(angle_first_layer_of_slab)))))
    return T

def calculate_transmittace_from_coeff_p_pol(t: ArrayLike,
                                            nk_first_layer_of_slab: ArrayLike,
                                            angle_first_layer_of_slab: ArrayLike,
                                            nk_last_layer_of_slab: ArrayLike,
                                            angle_last_layer_of_slab: ArrayLike) -> Array:
    """
    This function calculates the transmittance (T) for p-polarized light incident on a multilayer slab. 
    Transmittance is a measure of the fraction of incident light energy that passes through a material or structure. 
    In this function, we consider p-polarized light, which means the electric field vector is parallel to the plane of incidence.
    Physically, the transmittance quantifies the portion of light energy that successfully 
    passes through the slab structure, relative to the energy that initially interacted with it. The ratio calculated 
    incorporates the effects of refraction and wave impedance matching between the layers.

    **Arguments:**
    - `t`: A complex array-like object representing the transmission coefficient of p-polarized light. 
      This coefficient is derived from the scattering matrix formalism or similar optical computations and encapsulates
      the interaction of light with the layers of the slab.
    - `nk_first_layer_of_slab`: A complex array-like object representing the refractive index of the first layer of the slab. 
      The refractive index is a fundamental optical property that characterizes how light propagates through the medium.
    - `angle_first_layer_of_slab`: A real array-like object representing the angle of light incidence (or refraction) 
      in the first layer of the slab, in radians.
    - `nk_last_layer_of_slab`: A complex array-like object representing the refractive index of the last layer of the slab. 
      Similar to the first layer, this describes the propagation of light at the exit side of the slab.
    - `angle_last_layer_of_slab`: A real array-like object representing the angle of refraction in the last layer of the slab, in radians.

    **Returns:**
    - `T`: A real-valued array representing the transmittance (fraction of energy transmitted through the slab). 
      It is calculated using the ratio of energy flux in the outgoing medium to the energy flux in the incoming medium, 
      taking into account the transmission coefficient (`t`) and the refractive indices (`nk`) and angles (`angle`) 
      in the first and last layers.
    """
    T = jnp.multiply(jnp.square(jnp.abs(t)), jnp.true_divide(jnp.real(jnp.multiply(nk_last_layer_of_slab, jnp.conj(jnp.cos(angle_last_layer_of_slab)))),
                                                             jnp.real(jnp.multiply(nk_first_layer_of_slab, jnp.conj(jnp.cos(angle_first_layer_of_slab))))))
    return T

def calculate_transmittace_from_coeff(t: ArrayLike,
                                      nk_first_layer_of_slab: ArrayLike,
                                      angle_first_layer_of_slab: ArrayLike,
                                      nk_last_layer_of_slab: ArrayLike,
                                      angle_last_layer_of_slab: ArrayLike,
                                      polarization: ArrayLike) -> Array:
    """
    This function calculates the transmittance (T) of a system based on input parameters that describe the transmission coefficient (t), 
    the optical properties of the first and last layers of a slab, and the polarization of the incident light.
    This function acts as a selector for two sub-functions: `calculate_transmittace_from_coeff_s_pol` and 
    `calculate_transmittace_from_coeff_p_pol`. Based on the input `polarization`, it decides whether to calculate 
    the transmittance for 's' or 'p' polarization. The transmittance quantifies how much light is transmitted 
    through the slab system and depends on the optical properties and angles at the interfaces.

    Arguments:
    t: ArrayLike
        The transmission coefficient of the system, typically a complex value representing amplitude and phase information.

    nk_first_layer_of_slab: ArrayLike
        The complex refractive index of the first layer of the slab, which determines how light propagates through the layer. 
        It is a combination of the refractive index (n) and extinction coefficient (k) and may vary with wavelength.

    angle_first_layer_of_slab: ArrayLike
        The angle of incidence of the light on the first layer of the slab, typically given in radians. 
        This angle impacts how light refracts or reflects at the interface.

    nk_last_layer_of_slab: ArrayLike
        The complex refractive index of the last layer of the slab, which describes how the light exits the system.

    angle_last_layer_of_slab: ArrayLike
        The angle of transmission or refraction through the last layer of the slab. This is critical in determining the exit path of the light.

    polarization: bool
        A boolean indicating the polarization state of the light:
        - `False`: Indicates 's' polarization (electric field perpendicular to the plane of incidence).
        - `True`: Indicates 'p' polarization (electric field parallel to the plane of incidence).

    Returns:
    Array
        The transmittance (T) of the system, calculated based on the input polarization. The result is dependent on whether 
        the polarization is 's' or 'p', and the function redirects to the appropriate sub-function to perform the computation.
    """
    # Use `jnp.select` to determine which transmittance calculation to perform based on the polarization value.
    return jnp.select(condlist=[jnp.array_equal(polarization, jnp.array([False], dtype=bool)),
                                jnp.array_equal(polarization, jnp.array([True], dtype=bool))],
                    choicelist=[calculate_transmittace_from_coeff_s_pol(t,
                                                                        nk_first_layer_of_slab,
                                                                        angle_first_layer_of_slab,
                                                                        nk_last_layer_of_slab,
                                                                        angle_last_layer_of_slab),
                                calculate_transmittace_from_coeff_p_pol(t,
                                                                        nk_first_layer_of_slab,
                                                                        angle_first_layer_of_slab,
                                                                        nk_last_layer_of_slab,
                                                                        angle_last_layer_of_slab)])

# THE FUNCTIONS FOR OBTAINING THE COEFFICIENTS:

def compute_rt_at_interface_s(layer_idx: ArrayLike,
                              nk_angles_stack: ArrayLike) -> Array:
    """
    This function computes the reflection (r) and transmission (t) coefficients 
    for s-polarized light at the interface between two adjacent layers in a stack 
    of materials. 

    Arguments:
    - layer_idx: ArrayLike
        The index of the current layer in the stack. This is used to determine 
        which layers' refractive indices (n) and angles of incidence (theta) 
        are used in the calculation. Typically, this is an integer value that 
        corresponds to the position of the current layer in the material stack.

    - nk_angles_stack: ArrayLike
        A 2D array of shape [N, 2], where N is the number of layers in the stack. 
        Each row corresponds to a layer and contains:
          - [i, 0]: The refractive index (n) of the i-th layer.
          - [i, 1]: The angle of incidence (theta) of light in the i-th layer, 
                    measured with respect to the normal of the layer.

    Returns:
    - rt: Array
        An array containing the reflection (r) and transmission (t) coefficients 
        for s-polarized light at the interface between the specified layer (layer_idx) 
        and the subsequent layer (layer_idx + 1). These coefficients describe the 
        behavior of light as it transitions between the two layers.
    """
    # Extract the refractive index (n) of the first layer from nk_angles_stack at the specified layer index
    first_layer_n = nk_angles_stack.at[layer_idx, 0].get()
    
    # Extract the refractive index (n) of the second layer (layer_idx + 1) from nk_angles_stack
    second_layer_n = nk_angles_stack.at[jnp.add(layer_idx, jnp.array([1], dtype=jnp.int32)), 0].get()
    
    # Extract the angle of incidence (theta) of the first layer from nk_angles_stack at the specified layer index
    first_layer_theta = nk_angles_stack.at[layer_idx, 1].get()
    
    # Extract the angle of incidence (theta) of the second layer (layer_idx + 1) from nk_angles_stack
    second_layer_theta = nk_angles_stack.at[jnp.add(layer_idx, jnp.array([1], dtype=jnp.int32)), 1].get()
    
    # Compute the reflection and transmission coefficients for s-polarized light
    # between the two layers using the Fresnel equations.
    rt = fresnel_s(
        first_layer_n=first_layer_n,
        second_layer_n=second_layer_n,
        first_layer_theta=first_layer_theta,
        second_layer_theta=second_layer_theta
    )
    
    # Return the computed reflection and transmission coefficients as an array
    return rt

def compute_rt_at_interface_p(layer_idx: ArrayLike, 
                              nk_angles_stack: ArrayLike) -> Array:
    """
    This function calculates the reflection and transmission coefficients for p-polarized light at the 
    interface between two adjacent layers of different materials in a multilayer optical system. 
    The function calls `fresnel_p`, a helper function that uses the Fresnel equations for p-polarized light to 
    determine the reflection and transmission coefficients. These coefficients are crucial for modeling optical
    behavior in multilayer systems.

    Arguments:
    - layer_idx: ArrayLike
        The index of the current layer. It is used to access the optical properties (refractive index `n`) 
        and angles (relative to the layer normal) of the current layer and the adjacent layer.
        For example, if `layer_idx` is 2, the function retrieves the refractive indices and angles for 
        layer 2 and layer 3.

    - nk_angles_stack: ArrayLike
        A stack (2D array) containing the refractive indices (`n`) and angles for all layers. 
        The shape of this array is `[N, 2]`, where `N` is the number of layers.
        - `nk_angles_stack[i, 0]` provides the refractive index `n` for the i-th layer.
        - `nk_angles_stack[i, 1]` provides the angle of incidence (or refraction) of light 
          relative to the normal for the i-th layer.

    Returns:
    - Array
        The reflection and transmission coefficients for p-polarized light at the interface between the `layer_idx`-th layer and the `layer_idx + 1`-th layer. 
        These coefficients are computed using the Fresnel equations for p-polarized light.
    """
    
    # Retrieve the refractive index `n` of the first layer from `nk_angles_stack` using the given `layer_idx`.
    first_layer_n = nk_angles_stack.at[layer_idx, 0].get()
    
    # Retrieve the refractive index `n` of the second layer (next layer) by adding 1 to `layer_idx` and accessing `nk_angles_stack`.
    second_layer_n = nk_angles_stack.at[jnp.add(layer_idx, jnp.array([1], dtype=jnp.int32)), 0].get()
    
    # Retrieve the angle `theta` (relative to the normal) of the first layer from `nk_angles_stack` using `layer_idx`.
    first_layer_theta = nk_angles_stack.at[layer_idx, 1].get()
    
    # Retrieve the angle `theta` (relative to the normal) of the second layer (next layer) by adding 1 to `layer_idx` and accessing `nk_angles_stack`.
    second_layer_theta = nk_angles_stack.at[jnp.add(layer_idx, jnp.array([1], dtype=jnp.int32)), 1].get()
    
    # Compute the reflection and transmission coefficients for p-polarized light using the `fresnel_p` function.
    # The function is called with the refractive indices and angles of the first and second layers.
    rt = fresnel_p(first_layer_n=first_layer_n,
                   second_layer_n=second_layer_n,
                   first_layer_theta=first_layer_theta,
                   second_layer_theta=second_layer_theta)
    
    # Return the computed reflection and transmission coefficients.
    return rt

def vectorized_rt_s_pol():
    """
    The purpose of this function is to improve computational efficiency by vectorizing 
    the reflection and transmission coefficient calculations for s-polarized light. By 
    using JAX's `vmap`, the computations can process batched `index` values in mapped 
    way while ensuring that `nk_angles_stack` remains fixed.

    Arguments:
        None: This function does not take any direct input arguments. However, the returned 
              vectorized function expects the following arguments:
              - `index` (the first argument): A batched array representing the indices of the 
                nk_angles_stack array which is used for computation of  reflection and 
                transmission coefficients. This argument is vectorized along its first axis.
              - `nk_angles_stack` (the second argument): A non-vectorized array containing 
                the optical properties and angles of incidence that remain the same across all
                computations.

    Returns:
        Callable: A vectorized version of the `compute_rt_at_interface_s` function. This 
                  function applies the computations element-wise over the first axis of the 
                  `index` argument while keeping the `nk_angles_stack` argument constant.
    """
    return vmap(compute_rt_at_interface_s, (0, None))

def vectorized_rt_p_pol():
    """
    The purpose of this function is to improve computational efficiency by vectorizing 
    the reflection and transmission coefficient calculations for p-polarized light. By 
    using JAX's `vmap`, the computations can process batched `index` values in mapped 
    way while ensuring that `nk_angles_stack` remains fixed.

    Arguments:
        None: This function does not take any direct input arguments. However, the returned 
              vectorized function expects the following arguments:
              - `index` (the first argument): A batched array representing the indices of the 
                nk_angles_stack array which is used for computation of  reflection and 
                transmission coefficients. This argument is vectorized along its first axis.
              - `nk_angles_stack` (the second argument): A non-vectorized array containing 
                the optical properties and angles of incidence that remain the same across all
                computations.

    Returns:
        Callable: A vectorized version of the `compute_rt_at_interface_p` function. This 
                  function applies the computations element-wise over the first axis of the 
                  `index` argument while keeping the `nk_angles_stack` argument constant.
    """
    return vmap(compute_rt_at_interface_p, (0, None))

def polarization_based_rt_selection(layer_indices: ArrayLike, 
                                    nk_angles_stack: ArrayLike, 
                                    polarization: ArrayLike) -> Array:
    """ 
    This function selects the appropriate vectorized reflection and transmission calculation function 
    based on the polarization type (s or p-polarization).

    Arguments:
    layer_indices: 
        - This is an input array of indices specifying the layers in a multilayer thin film. It is an 
          integer array (of type int32) that indicates the specific layers over which the r and t 
          calculations are to be performed.
        
    nk_angles_stack:
        - This is an array containing the refractive index (n) and extinction coefficient (k) values 
          for each layer. It also includes the angle between the layer normal and the light ray for 
          each layer. This stack is used in the RT calculations to determine how the light propagates
          through each layer.
        
    polarization:
        - This is a boolean array, which dictates the type of polarization for the calculation:
            - False indicates s-polarization (perpendicular polarization).
            - True indicates p-polarization (parallel polarization).

    Returns:
        Array: This function returns an array of reflection and transmission coefficients calculated using 
            either the s-polarization or p-polarization. Depending on the polarization input, 
            it calls different vectorized rt functions.
    """

    return jnp.select(condlist=[jnp.array_equal(polarization, jnp.array([False], dtype=bool)),
                                jnp.array_equal(polarization, jnp.array([True], dtype=bool))],
                    choicelist=[vectorized_rt_s_pol()(layer_indices, nk_angles_stack),
                                vectorized_rt_p_pol()(layer_indices, nk_angles_stack)])

def compute_rt(nk_list: ArrayLike, angles: ArrayLike, polarization: ArrayLike) -> Array:
    """
    This function computes the reflectance (rt) for a multilayer structure. 
    It takes in three arguments:
    
    - nk_list: This is a list (or array) containing the complex refractive index values for the layers in the multilayer stack. 
              The complex refractive index is typically represented as `n + ik`, where `n` is the real refractive index and `k` 
              is the extinction coefficient. The list `nk_list` contains the refractive index values for all N+2 media in the 
              stack structure, including the two outermost media (such as air and substrate).
    
    - angles: This array contains the incident angles (the angle between the normal to the surface and the incoming light) for
              each medium in the multilayer stack. The angles correspond to the light's propagation direction in each layer of 
              the stack.
    
    - polarization: This is a boolean flag (True or False). If `polarization` is False, the function will use the s-polarization (perpendicular polarization) in calculations, 
                    and if `True`, it will use the p-polarization (parallel polarization). This flag determines how the reflectance (rt) is computed 
                    based on the light's polarization state.
    """
    
    # Concatenate the complex refractive indices and angles into a stack (a 2D array).
    # `nk_list` is reshaped to add an extra dimension (axis 1) and then concatenated with `angles`, 
    # creating a 2D stack of both `nk` values (refractive index and extinction coefficient) and angles.
    nk_angles_stack = jnp.concat([jnp.expand_dims(nk_list, 1), jnp.expand_dims(angles, 1)], axis=1)
    
    # Compute the stop value, which is the size of `nk_list` minus 1, to define the last layer in the multilayer stack.
    stop_value = int(jnp.size(nk_list)) - 1
    
    # Generate a 1D array `layer_indices` that contains the indices of the layers, from 0 to `stop_value-1`.
    # This is used for iterating over each layer in the multilayer stack structure.
    layer_indices = jnp.arange(stop_value, dtype=jnp.int32)

    # Call another function `polarization_based_rt_selection` with the `layer_indices`, `nk_angles_stack`, and `polarization`
    # to compute the reflectance (rt). This function likely takes care of the specific calculations based on polarization 
    # and uses the layer information to compute the reflectance at each interface in the stack.
    return polarization_based_rt_selection(layer_indices, nk_angles_stack, polarization)

def compute_r_t_magnitudes_incoh(coherency_index: ArrayLike, rts: ArrayLike, nk_list: ArrayLike, layer_angles: ArrayLike, polarization: ArrayLike) -> Array:
    R = calculate_reflectance_from_coeff(rts.at[coherency_index,0].get())
    T = calculate_transmittace_from_coeff(t = rts.at[coherency_index,1].get(),
                                          nk_first_layer_of_slab = nk_list.at[coherency_index].get(),
                                          angle_first_layer_of_slab = layer_angles.at[coherency_index].get(),
                                          nk_last_layer_of_slab = nk_list.at[coherency_index+1].get(),
                                          angle_last_layer_of_slab = layer_angles.at[coherency_index+1].get(),
                                          polarization = polarization)
    return R,T

def compute_incoh_r_t_magnitudes_incoh_film(coherency_index: ArrayLike, rts: ArrayLike, nk_list: ArrayLike, layer_angles: ArrayLike, polarization: ArrayLike):
    """
    This function calculates the reflection and transmission magnitudes for an incoherent layer in the multilayer optical thin films.

    Purpose:
    It computes the reflection (R) and transmission (T) magnitudes of an incoherent multilayer optical thin film 
    using precomputed reflection and transmission coefficients (`rts`), along with refractive indices (`nk_list`), 
    layer angles (`layer_angles`), and polarization states.

    Arguments:
    - coherency_index (ArrayLike): Index representing the specific layer in the incoherent multilayer structure.
    - rts (ArrayLike): An array containing reflection and transmission coefficients for different layers.
    - nk_list (ArrayLike): An array containing the complex refractive indices of the layers.
    - layer_angles (ArrayLike): An array containing the angles of incidence or refraction for each layer.
    - polarization (ArrayLike): An array indicating the polarization state of the light (s- or p-polarized).

    Returns:
    - R,T (Tuple): A tuple containing the calculated reflection magnitude (R) and transmission magnitude (T).
    """

    # Extract and compute the reflection magnitude from the reflection coefficient
    R = calculate_reflectance_from_coeff(rts.at[coherency_index, 0].get())

    # Compute the transmission magnitude using the transmission coefficient 
    # and the refractive indices and angles of the first and last layers in the slab.
    T = calculate_transmittace_from_coeff(
        t = rts.at[coherency_index, 1].get(),  # Transmission coefficient for the current layer
        nk_first_layer_of_slab = nk_list.at[coherency_index].get(),  # Refractive index of the first layer
        angle_first_layer_of_slab = layer_angles.at[coherency_index].get(),  # Angle of the first layer
        nk_last_layer_of_slab = nk_list.at[coherency_index + 1].get(),  # Refractive index of the last layer
        angle_last_layer_of_slab = layer_angles.at[coherency_index + 1].get(),  # Angle of the last layer
        polarization = polarization  # Polarization state of the light 0 or 1 represents 's' or 'p'
    )

    # Return calculated reflection and transmission magnitudes
    return R, T

def compute_coh_r_t_magnitudes_incoh_film(coherency_index: ArrayLike, coherent_layer_indices: ArrayLike, rts: ArrayLike, 
                                          layer_phases: ArrayLike, nk_list: ArrayLike, layer_angles: ArrayLike, polarization: ArrayLike):
    """
    Computes the reflection (R) and transmission (T) magnitudes of the coherent layers 
    in an incoherent multilayer optical thin film. The coherent layer can consist of 
    one or more adjacent coherent stacks.
    
    Arguments:
        coherency_index (ArrayLike): Index representing the position of the coherent stack within the multilayer film.
        coherent_layer_indices (ArrayLike): A 2D array where each row defines the start and end indices 
                                           of a coherent stack in the multilayer film.
        rts (ArrayLike): A 2D array containing reflection and transmission coefficients for all layers.
        layer_phases (ArrayLike): Phase shifts corresponding to each layer in the multilayer system.
        nk_list (ArrayLike): A list of complex refractive indices for each layer.
        layer_angles (ArrayLike): A list of angles of incidence/refraction for each layer.
        polarization (ArrayLike): Polarization state of the incident light (TE or TM mode).
    
    Returns:
        Tuple (R, T):
            R (float): Reflectance of the coherent stack.
            T (float): Transmittance of the coherent stack.
    """
    # Determine the first and last indices of the coherent stack within the multilayer system
    start_index = coherent_layer_indices.at[coherency_index-1, 0].get() - 1
    last_index = coherent_layer_indices.at[coherency_index-1, 1].get() + 1
    
    # Extract the reflection and transmission coefficients for the coherent stack
    stack_rts = rts.at[start_index:last_index, :].get()
    
    # Extract the phase shifts corresponding to each layer in the coherent stack
    stack_layer_phases = layer_phases.at[start_index:last_index-1].get()
    
    # Perform matrix multiplication to obtain the cascaded transfer matrix for the coherent stack
    tr_matrix = coh_cascaded_matrix_multiplication(phases=stack_layer_phases, rts=stack_rts.at[1:,:].get())
    
    # Multiply the initial transmission matrix with the cascaded transfer matrix
    tr_matrix = jnp.multiply(jnp.true_divide(1, stack_rts.at[0,1].get()), 
                              jnp.dot(jnp.array([[1, stack_rts.at[0,0].get()], 
                                                 [stack_rts.at[0,0].get(), 1]]), 
                                     tr_matrix))
    
    # Compute the reflection coefficient
    r = jnp.true_divide(tr_matrix.at[1,0].get(), tr_matrix.at[0,0].get())
    
    # Compute the transmission coefficient
    t = jnp.true_divide(1, tr_matrix.at[0,0].get())
    
    # Compute the reflectance from the reflection coefficient
    R = calculate_reflectance_from_coeff(r)
    
    # Compute the transmittance from the transmission coefficient, considering material properties and polarization
    T = calculate_transmittace_from_coeff(t,
                                          nk_list.at[start_index].get(),
                                          layer_angles.at[start_index].get(),
                                          nk_list.at[last_index].get(),
                                          layer_angles.at[last_index].get(),
                                          polarization)
    
    return R, T

def compute_r_t_magnitudes_incoh(coherent_layer_indices: ArrayLike, coherency_indices: ArrayLike, rts: ArrayLike, 
                                 layer_phases: ArrayLike, nk_list: ArrayLike, layer_angles: ArrayLike, polarization: ArrayLike) -> Array:
    layer_magnitudes = jnp.zeros((len(coherency_indices)-1, 2))

    for i in range(len(coherency_indices)-1):
        coherency_index = coherency_indices.at[i].get()
        condition = jnp.greater(coherency_index, -1)
        coherency_index = jnp.abs(coherency_index)
        if condition:
            R,T = compute_coh_r_t_magnitudes_incoh_film(coherency_index, rts, nk_list, layer_angles, polarization)
        else:
            R,T = compute_coh_r_t_magnitudes_incoh_film(coherency_index, coherent_layer_indices, rts, layer_phases, nk_list, layer_angles, polarization)

        layer_magnitudes = layer_magnitudes.at[i,0].set(R)
        layer_magnitudes = layer_magnitudes.at[i,1].set(T)

    return layer_magnitudes

def compute_r_t_magnitudes_incoh(coherent_layer_indices: ArrayLike, coherency_indices: ArrayLike, rts: ArrayLike, 
                                 layer_phases: ArrayLike, nk_list: ArrayLike, layer_angles: ArrayLike, polarization: ArrayLike) -> Array:
    """
    This function computes the reflection (R) and transmission (T) magnitudes for each layer in an incoherent multilayer optical thin film. 
    It iterates through the layers of a multilayer optical thin film and calculates the reflection and transmission magnitudes for both coherent 
    and incoherent layers and stacks.

    Arguments:
    - coherent_layer_indices (ArrayLike): Indices of the coherent layers in the multilayer system.
    - coherency_indices (ArrayLike): Indices indicating whether a layer is coherent or incoherent.
    - rts (ArrayLike): Reflection and transmission coefficients for the layers.
    - layer_phases (ArrayLike): Phase shifts of the layers due to optical thickness.
    - nk_list (ArrayLike): Refractive index (n + jk) values of the materials in the multilayer system.
    - layer_angles (ArrayLike): Incident angles for each layer.
    - polarization (ArrayLike): Polarization states of the incident light.

    Returns:
    - layer_magnitudes (Array): A 2D array where each row contains the reflection (R) and transmission (T) magnitudes for a layer.
    """

    # Initialize an array to store reflection (R) and transmission (T) magnitudes for each layer
    layer_magnitudes = jnp.zeros((len(coherency_indices) - 1, 2))

    # Iterate over all layers except the last one
    for i in range(len(coherency_indices) - 1):
        # Get the coherency index for the current layer
        coherency_index = coherency_indices.at[i].get()

        # Check if the layer is coherent (coherency_index > -1)
        condition = jnp.greater(coherency_index, -1)

        # Convert the coherency index to an absolute value
        coherency_index = jnp.abs(coherency_index)

        if condition:
            # If the layer is coherent, compute R and T using the appropriate function
            R, T = compute_coh_r_t_magnitudes_incoh_film(coherency_index, rts, nk_list, layer_angles, polarization)
        else:
            # If the layer is incoherent, use an alternative computation including phase information
            R, T = compute_coh_r_t_magnitudes_incoh_film(coherency_index, coherent_layer_indices, rts, layer_phases, nk_list, layer_angles, polarization)

        # Store the computed reflection and transmission magnitudes in the result array
        layer_magnitudes = layer_magnitudes.at[i, 0].set(R)
        layer_magnitudes = layer_magnitudes.at[i, 1].set(T)

    # Return the computed reflection and transmission magnitudes for all layers
    return layer_magnitudes