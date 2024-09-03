from typing import Optional, Union, Tuple
import jax
import jax.numpy as jnp
from jax import vmap, jit

from katmer.stacks import Stack
from katmer.light import Light


def _matmul(carry, phase_t_r):
    """
    Multiplies two complex matrices in a sequence.
    
    Args:
        carry (jax.numpy.ndarray): The accumulated product of the matrices so far.
                                   This is a 2x2 complex matrix.
        phase_t_r (jax.numpy.ndarray): phase = e^-i delta or  |e^-i delta|^2, t (or T) and r (or R)
        
    Returns:
        jax.numpy.ndarray: The updated product after multiplying the carry with the current matrix.
                           This is also a 2x2 complex matrix.
        None: A placeholder required by jax.lax.scan for compatibility.
    """
    mat = jnp.array(1/phase_t_r[1]) * jnp.dot(jnp.array([[phase_t_r[0], 0],
                                                         [0, phase_t_r[0]]]), 
                                              jnp.array([[1, phase_t_r[2]],
                                                         [phase_t_r[2], 1]]))
    return jnp.dot(carry, mat), None  # Perform matrix multiplication and return the result.

def _cascaded_matrix_multiplication(phases_ts_rs):
    """
    Performs cascaded matrix multiplication on a sequence of complex matrices using scan.
    
    Args:
        matrices (jax.numpy.ndarray): An array of shape [N, 2, 2], where N is the number of 2x2 complex matrices.
        
    Returns:
        jax.numpy.ndarray: The final result of multiplying all the matrices together in sequence.
                           This is a single 2x2 complex matrix.
    """
    initial_value = jnp.eye(2, dtype=jnp.complex64)  # Start with the identity matrix of size 2x2.
    result, _ = jax.lax.scan(_matmul, initial_value, phases_ts_rs)  # Accumulate the product over all matrices.
    return result  # Return the final accumulated product.

def _create_phases_ts_rs(_trs, _phases, polarization):
    N = _phases.shape[0]  # Get the dimension N

    if polarization is None:
        def process_element(i):
            return (_phases[i], _trs[i][0, 0], _trs[i][0, 1], _trs[i][1, 0], _trs[i][1, 1])
    else:
        def process_element(i):
            return [_phases[i], _trs[i][0], _trs[i][1]]

    # Apply process_element function across all indices
    result = jax.vmap(process_element)(jnp.arange(N))
    return result


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

@jit
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
    _theta_indices = jnp.arange(0,jnp.size(light.theta), dtype = int) # Array or single value for the indices of angle of incidence
    _wavelength_indices = jnp.arange(0,jnp.size(light.wavelength), dtype = int) # Array or single value for  the indices of wavelength

    # Vectorize the _tmm function across theta and wavelength using JAX's vmap
    _tmm_vectorized = vmap(
        _tmm, 
        in_axes=(None, None, 0, 0)  # Fix _stack and _polarization, vmap _theta_indices and _wavelength_indices
    )

    # Apply the vectorized function to the theta and wavelength arrays
    _result = _tmm_vectorized(stack, light, _polarization, _theta_indices, _wavelength_indices)

    # Return the result
    return _result
