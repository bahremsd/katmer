from typing import Optional, Union, Tuple
import jax
import jax.numpy as jnp
from jax import vmap, jit

from katmer.stacks import Stack
from katmer.light import Light


import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
from typing import Union, List

def _matmul(carry, phase_t_r):
    """
    Multiplies two complex matrices in a sequence.

    Args:
        carry (jax.numpy.ndarray): The accumulated product of the matrices so far.
                                   This is expected to be a 2x2 complex matrix.
        phase_t_r (jax.numpy.ndarray): A 3-element array where:
            - phase_t_r[0] represents the phase shift delta (a scalar).
            - phase_t_r[1] represents the transmission coefficient t or T (a scalar).
            - phase_t_r[2] represents the reflection coefficient r or R (a scalar).

    Returns:
        jax.numpy.ndarray: The updated product after multiplying the carry matrix with the current matrix.
                           This is also a 2x2 complex matrix.
        None: A placeholder required by jax.lax.scan for compatibility.
    """
    # Create the diagonal phase matrix based on phase_t_r[0]
    # This matrix introduces a phase shift based on the delta value
    phase_matrix = jnp.array([[jnp.exp(-1j * phase_t_r[0]), 0],  # Matrix with phase shift for the first entry
                              [0, jnp.exp(1j * phase_t_r[0])]])  # Matrix with phase shift for the second entry

    # Create the matrix based on phase_t_r[1] and phase_t_r[2]
    # This matrix incorporates the transmission and reflection coefficients
    transmission_reflection_matrix = jnp.array([[1, phase_t_r[1]],  # Top row with transmission coefficient
                                               [phase_t_r[1], 1]])  # Bottom row with transmission coefficient

    # Compute the current matrix by multiplying the phase_matrix with the transmission_reflection_matrix
    # The multiplication is scaled by 1/phase_t_r[2] to account for the reflection coefficient
    mat = jnp.array(1 / phase_t_r[2]) * jnp.dot(phase_matrix, transmission_reflection_matrix)

    # Multiply the accumulated carry matrix with the current matrix
    # This updates the product with the new matrix
    result = jnp.dot(carry, mat)

    return result, None  # Return the updated matrix and None as a placeholder for jax.lax.scan

def _cascaded_matrix_multiplication(phases_ts_rs: jnp.ndarray) -> jnp.ndarray:
    """
    Performs cascaded matrix multiplication on a sequence of complex matrices using scan.

    Args:
        phases_ts_rs (jax.numpy.ndarray): An array of shape [N, 2, 2], where N is the number of 2x2 complex matrices.
                                          Each 2x2 matrix is represented by its 2x2 elements arranged in a 3D array.

    Returns:
        jax.numpy.ndarray: The final result of multiplying all the matrices together in sequence.
                           This result is a single 2x2 complex matrix representing the accumulated product of all input matrices.
    """
    initial_value = jnp.eye(2, dtype=jnp.complex128)  
    # Initialize with the identity matrix of size 2x2. # The identity matrix acts as the multiplicative identity, 
    # ensuring that the multiplication starts correctly.

    # jax.lax.scan applies a function across the sequence of matrices. 
    #Here, _matmul is the function applied, starting with the identity matrix.
    # `result` will hold the final matrix after processing all input matrices.
    result, _ = jax.lax.scan(_matmul, initial_value, phases_ts_rs)  # Scan function accumulates results of _matmul over the matrices.

    return result  # Return the final accumulated matrix product. # The result is the product of all input matrices in the given sequence.


def _create_phases_ts_rs(_trs: jnp.ndarray, _phases: jnp.ndarray) -> jnp.ndarray:
    """
    Create a new array combining phase and ts values.

    Args:
        _trs (jnp.ndarray): A 2D array of shape (N, 2) where N is the number of elements. 
                            Each element is a pair of values [t, s].
        _phases (jnp.ndarray): A 1D array of shape (N,) containing phase values for each element.

    Returns:
        jnp.ndarray: A 2D array of shape (N, 3) where each row is [phase, t, s].
                     The phase is from _phases, and t, s are from _trs.
    """

    N = _phases.shape[0]  # Get the number of elements (N) in the _phases array

    def process_element(i: int) -> List[float]:
        """
        Process an individual element to create a list of phase and ts values.

        Args:
            i (int): Index of the element to process.

        Returns:
            List[float]: A list containing [phase, t, s] where:
                - phase: The phase value from _phases at index i
                - t: The first value of the pair in _trs at index i
                - s: The second value of the pair in _trs at index i
        """
        return [_phases[i], _trs[i][0], _trs[i][1]]  # Return the phase and ts values as a list

    # Apply process_element function across all indices from 0 to N-1
    result = jax.vmap(process_element)(jnp.arange(N))  # jax.vmap vectorizes the process_element function
                                                    # to apply it across all indices efficiently
    
    return result  # Return the result as a 2D array of shape (N, 3)


def _tmm(stack: Stack, light: Light,
         theta_index: Union[int, jnp.ndarray],
         wavelength_index: Union[int, jnp.ndarray]
        ) -> Union[
            Tuple[jnp.ndarray, jnp.ndarray],
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:

    """
    The _tmm function computes optical properties such as reflectance (R), transmittance (T), and optionally absorbed energy,
    ellipsometric data (psi, delta), and the Poynting vector using the Transfer Matrix Method (TMM). The function is designed 
    for advanced simulations involving multilayer thin films, accommodating various material properties, polarization states,
    and conditions (e.g., incoherent layers).

    This function is intended to be used within another function, hence the underscore prefix, and it is optimized for JAX,
    enabling high-performance, differentiable computations. The function is versatile, supporting both scalar and array 
    inputs for the angle of incidence (theta) and wavelength of light.

    Args:
        stack (Stack): An object representing the multilayer thin film stack. This class includes the necessary methods 
                       and properties for material properties, layer thicknesses, and the handling of coherent and incoherent 
                       layers.
        polarization (Optional[Union[str, bool]]): The polarization state of the incident light. 's' denotes s-polarization, 
                                                  'p' denotes p-polarization, and None indicates unpolarized light.
        theta_index (Union[float, jnp.ndarray]): The index of angle of incidence (in radians) at which light strikes the multilayer thin 
                                           film. This can be a single int value or a jnp array for angle-dependent 
                                           computations.
        wavelength_index (Union[float, jnp.ndarray]): The index of wavelength(s) of the incident light. This can be a single int value 
                                                or a jnp array to compute properties over a range of wavelengths.

    Returns:
        Union[Tuple[jnp.ndarray, jnp.ndarray], 
              Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], 
              Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], 
              Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
              Depending on the state of the Stack object's flags, the function returns a tuple containing arrays of 
              reflectance (R), transmittance (T), and optionally absorbed energy, ellipsometric data (psi, delta), 
              and the Poynting vector.
    """
    # Helper function to calculate reflectance (R) and transmittance (T)
    def calculate_rt():
        """
        This helper function computes the reflection and transmission coefficients, denoted as R and T, for the multilayer 
        thin film. The calculation is performed using the Transfer Matrix Method (TMM), which involves constructing 
        characteristic matrices for each layer, considering the angle of incidence, wavelength, and polarization of the light.
        """
        layer_phases = stack.kz[theta_index, wavelength_index, :] * stack.thicknesses
        _phases_ts_rs = _create_phases_ts_rs(stack.rt[theta_index, wavelength_index, :], layer_phases, light.polarization)
        _tr_matrix = _cascaded_matrix_multiplication(_phases_ts_rs)
        R = _tr_matrix[1,0] / _tr_matrix[0,0]
        T = 1 / _tr_matrix[0,0]
        return R, T

    # Helper function to calculate absorbed energy
    def calculate_absorbed_energy(stack, r, t):
        """
        This helper function calculates the absorbed energy within the multilayer thin film structure. The calculation is 
        dependent on the complex reflection (r) and transmission (t) coefficients, which provide information about how much 
        energy is absorbed within each layer.
        """
        # Perform calculations based on r and t to determine absorbed energy
        # ...
        return absorbed_energy

    # Helper function to calculate ellipsometric data (psi, delta)
    def calculate_ellipsometric_data(stack, r, t):
        """
        This helper function computes the ellipsometric parameters psi and delta, which describe the change in polarization 
        state as light reflects off the multilayer thin film. The calculations are based on the complex reflection coefficients 
        (r) for different polarizations.
        """
        # Calculate psi and delta from r and t for coherent layers
        # ...
        return psi, delta

    # Helper function to calculate the Poynting vector
    def calculate_poynting_vector(stack, r, t):
        """
        The Poynting vector represents the directional energy flux (the rate of energy transfer per unit area) of the 
        electromagnetic wave. This helper function calculates the Poynting vector based on the complex transmission coefficient 
        (t) and the structure of the multilayer thin film.
        """
        # Calculate the Poynting vector based on t
        # ...
        return poynting_vector

    R, T = calculate_rt()
    results = [R, T]
    
    if not stack.any_incoherent:

        if stack.obs_absorbed_energy:
            absorbed_energy = calculate_absorbed_energy(stack, R, T)
            results.append(absorbed_energy)

        if stack.obs_ellipsiometric:
            psi, delta = calculate_ellipsometric_data(stack, R, T)
            results.append([psi, delta])

        if stack.obs_poynting:
            poynting_vector = calculate_poynting_vector(stack, R, T)
            results.append(poynting_vector)

        return tuple(results)


def forward(stack: Stack, light: Light) -> Union[
        jnp.ndarray, 
        tuple[jnp.ndarray, jnp.ndarray], 
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], 
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], 
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    The `forward` function applies the Transfer Matrix Method (TMM) over a range of wavelengths and angles of incidence (theta)
    by leveraging the JAX `vmap` function for vectorized computation. It is designed to handle multilayer thin film simulations
    efficiently, processing the entire spectrum and angular range of interest in one go.

    This function is highly optimized for performance and uses advanced Python techniques to ensure the calculations are both
    efficient and scalable. It does not include JIT compilation, as this might be applied externally.

    Args:
        _stack (Stack): The `Stack` object representing the multilayer thin film configuration. It includes material properties,
                       layer thicknesses, and options for incoherent layer handling.
        _light (Light): The `Light` object containing the properties of the incident light, including wavelength, angle of 
                      incidence (theta), and polarization.

    Returns:
        Union[jnp.ndarray, 
              tuple[jnp.ndarray, jnp.ndarray], 
              tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], 
              tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], 
              tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
              The output of the `_tmm` function, vectorized over the provided range of wavelengths and angles.
              Depending on the configuration of the `Stack` object, it may return:
              - Reflectance (R) and Transmittance (T) as `jnp.ndarray`
              - Optionally, absorbed energy, ellipsometric data (psi, delta), and the Poynting vector.
    """

    # Extract polarization, theta, and wavelength from the Light object
    _polarization = light.polarization  # 's', 'p', or None (unpolarized)
    _theta_indices = jnp.arange(0,len(light.angle_of_incidence), dtype = int) # Array or single value for the indices of angle of incidence
    _wavelength_indices = jnp.arange(0,len(light.wavelength), dtype = int) # Array or single value for  the indices of wavelength

    # Vectorize the _tmm function across theta and wavelength using JAX's vmap
    _tmm_vectorized = vmap(vmap(_tmm, (None, None, 0, None)), (None, None, None, 0)) # Fix _stack and _polarization, vmap _theta_indices and _wavelength_indices

    # Apply the vectorized function to the theta and wavelength arrays
    _result = _tmm_vectorized(stack, light, _theta_indices, _wavelength_indices)

    # Return the result
    return _result
