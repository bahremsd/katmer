from typing import Optional, Union, Tuple
import jax
import jax.numpy as jnp
from jax import vmap, jit

from stacks import Stack
from light import Light


def _matmul(carry, mat):
    """
    Multiplies two complex matrices in a sequence.
    
    Args:
        carry (jax.numpy.ndarray): The accumulated product of the matrices so far.
                                   This is a 2x2 complex matrix.
        mat (jax.numpy.ndarray): The current 2x2 complex matrix to multiply with the carry.
        
    Returns:
        jax.numpy.ndarray: The updated product after multiplying the carry with the current matrix.
                           This is also a 2x2 complex matrix.
        None: A placeholder required by jax.lax.scan for compatibility.
    """
    return jnp.dot(carry, mat), None  # Perform matrix multiplication and return the result.


def _cascaded_matrix_multiplication(matrices):
    """
    Performs cascaded matrix multiplication on a sequence of complex matrices using scan.
    
    Args:
        matrices (jax.numpy.ndarray): An array of shape [N, 2, 2], where N is the number of 2x2 complex matrices.
        
    Returns:
        jax.numpy.ndarray: The final result of multiplying all the matrices together in sequence.
                           This is a single 2x2 complex matrix.
    """
    initial_value = jnp.eye(2, dtype=jnp.complex64)  # Start with the identity matrix of size 2x2.
    result, _ = jax.lax.scan(_matmul, initial_value, matrices)  # Accumulate the product over all matrices.
    return result  # Return the final accumulated product.

def _fresnel_s(_first_layer_n: Union[float, jnp.ndarray], _second_layer_n: Union[float, jnp.ndarray],
               _first_layer_theta: Union[float, jnp.ndarray], _second_layer_theta: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the Fresnel reflection and transmission coefficients for s-polarized light.

    This function calculates the reflection and transmission coefficients for s-polarized light, where
    the electric field is perpendicular to the plane of incidence. The coefficients are computed based
    on the Fresnel equations, which describe the behavior of electromagnetic waves at the interface between
    two media with different refractive indices (n&k).

    Parameters:
    - _first_layer_n (Union[float, jnp.ndarray]): The refractive index of the first medium. This can be
      either a scalar or an array, allowing for flexible and vectorized operations.
    - _second_layer_n (Union[float, jnp.ndarray]): The refractive index of the second medium. This can
      also be a scalar or an array, consistent with the first layer's refractive index.
    - _first_layer_theta (Union[float, jnp.ndarray]): The angle of incidence in the first medium, provided
      as either a scalar or an array.
    - _second_layer_theta (Union[float, jnp.ndarray]): The angle of refraction in the second medium, given
      as either a scalar or an array.

    Returns:
    - Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
        - _r_s (jnp.ndarray): The Fresnel reflection coefficient for s-polarized light, representing the ratio
          of the reflected electric field amplitude to the incident electric field amplitude.
        - _t_s (jnp.ndarray): The Fresnel transmission coefficient for s-polarized light, representing the ratio
          of the transmitted electric field amplitude to the incident electric field amplitude.

    The reflection coefficient (_r_s) and transmission coefficient (_t_s) are computed using the following
    Fresnel equations:
    
    - Reflection coefficient (_r_s):
      _r_s = (n_i cos \theta_i - n_i+1 cos \theta_i+1) / (n_i cos \theta_i + n_i+1 cos \theta_i+1)
    
    - Transmission coefficient (_t_s):
      _t_s = ( 2 * n_i cos \theta_i) / (n_i cos \theta_i + n_i+1 cos \theta_i+1)
    
    """
    _r_s = ((_first_layer_n * jnp.cos(_first_layer_theta) - _second_layer_n * jnp.cos(_second_layer_theta)) /
            (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))
    _t_s = (2 * _first_layer_n * jnp.cos(_first_layer_theta) /
            (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))
    return _r_s, _t_s

def _fresnel_p(_first_layer_n: Union[float, jnp.ndarray], _second_layer_n: Union[float, jnp.ndarray],
               _first_layer_theta: Union[float, jnp.ndarray], _second_layer_theta: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This function computes the reflection and transmission coefficients for light polarized with the
    electric field parallel to the plane of incidence (p-polarization). The coefficients are derived
    from the Fresnel equations, which describe the interaction of electromagnetic waves with an interface
    between two media with different refractive indices (n&k).

    Parameters:
    - _first_layer_n (Union[float, jnp.ndarray]): The refractive index of the first medium. This parameter
      can be a single scalar value or a JAX array, allowing for operations on multiple values simultaneously.
    - _second_layer_n (Union[float, jnp.ndarray]): The refractive index of the second medium, provided as
      either a scalar or an array.
    - _first_layer_theta (Union[float, jnp.ndarray]): The angle of incidence in the first medium. Can be
      a scalar or an array of angles.
    - _second_layer_theta (Union[float, jnp.ndarray]): The angle of refraction in the second medium. This
      can also be provided as a scalar or an array.

    Returns:
    - Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
      - _r_p (jnp.ndarray): The Fresnel reflection coefficient for p-polarized light. This coefficient
        represents the fraction of the incident light that is reflected at the interface between the two
        media.
      - _t_p (jnp.ndarray): The Fresnel transmission coefficient for p-polarized light. This coefficient
        represents the fraction of the incident light that is transmitted through the interface.

    The reflection (_r_p) and transmission (_t_p) coefficients are computed using the following equations:
    - Reflection coefficient (_r_p):
      _r_p = (n_i+1 cos \theta_i - n_i cos \theta_i+1) / (n_i+1 cos \theta_i + n_i cos \theta_i+1)
      
    - Transmission coefficient (_t_p):
      _t_p = ( 2 * n_i cos \theta_i) / (n_i+1 cos \theta_i + n_i cos \theta_i+1)
      
    """
    _r_p = ((_second_layer_n * jnp.cos(_first_layer_theta) - _first_layer_n * jnp.cos(_second_layer_theta)) /
            (_second_layer_n * jnp.cos(_first_layer_theta) + _first_layer_n * jnp.cos(_second_layer_theta)))
    _t_p = (2 * _first_layer_n * jnp.cos(_first_layer_theta) /
            (_second_layer_n * jnp.cos(_first_layer_theta) + _first_layer_n * jnp.cos(_second_layer_theta)))
    return _r_p, _t_p

def _interface(
    _any_incoherent: bool,
    _polarization: bool,
    _first_layer_theta: Union[float, jnp.ndarray],
    _second_layer_theta: Union[float, jnp.ndarray],
    _first_layer_n: Union[float, jnp.ndarray],
    _second_layer_n: Union[float, jnp.ndarray]
) -> Union[jnp.ndarray, Tuple[float, float]]:
    """
    Calculate reflectance and transmittance at the interface between two layers
    using Fresnel equations, considering the polarization of light and whether
    the layers are coherent or incoherent.

    Parameters
    ----------
    _any_incoherent : bool
        Flag indicating if any of the layers at the boundary are incoherent.
        - False: Both boundaries are coherent.
        - True: At least one boundary is incoherent.
    
    _polarization : Optional[bool]
        Polarization state of the incoming light.
        - None: Both s and p polarizations (unpolarized light).
        - False: s-polarization.
        - True: p-polarization.
    
    _first_layer_theta : Union[float, jnp.ndarray]
        Angle of incidence with respect to the normal at the boundary of the first layer.
    
    _second_layer_theta : Union[float, jnp.ndarray]
        Angle of refraction with respect to the normal at the boundary of the second layer.
    
    _first_layer_n : Union[float, jnp.ndarray]
        Refractive index of the first layer.
    
    _second_layer_n : Union[float, jnp.ndarray]
        Refractive index of the second layer.
    
    Returns
    -------
    Union[jnp.ndarray, Tuple[float, float]]
        - If polarization is None (both s and p), return a 2x2 matrix:
          - Coherent: [[r_s, r_p], [t_s, t_p]]
          - Incoherent: [[R_s, R_p], [T_s, T_p]]
        - If polarization is False, return a vector [r_s, t_s] or [R_s, T_s].
        - If polarization is True, return a vector [r_p, t_p] or [R_p, T_p].
    """
    # Handle the incoherent case
    if _any_incoherent:
        if _polarization is None:
            # Unpolarized light: both s and p polarizations
            _r_s, _t_s = _fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            _r_p, _t_p = _fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            _R_s = jnp.abs(_r_s) ** 2
            _T_s = jnp.abs(_t_s) ** 2
            _R_p = jnp.abs(_r_p) ** 2
            _T_p = jnp.abs(_t_p) ** 2
            return jnp.array([[_R_s, _R_p], [_T_s, _T_p]])
        elif _polarization is False:
            _r_s, _t_s = _fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            _R_s = jnp.abs(_r_s) ** 2
            _T_s = jnp.abs(_t_s) ** 2
            return jnp.array([_R_s, _T_s])
        elif _polarization is True:
            _r_p, _t_p = _fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            _R_p = jnp.abs(_r_p) ** 2
            _T_p = jnp.abs(_t_p) ** 2
            return jnp.array([_R_p, _T_p])

    # Handle the coherent case
    else:
        if _polarization is None:
            # Unpolarized light: both s and p polarizations
            _r_s, _t_s = _fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            _r_p, _t_p = _fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            return jnp.array([[_r_s, _r_p], [_t_s, _t_p]])
        elif _polarization is False:
            _r_s, _t_s = _fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            return jnp.array([_r_s, _t_s])
        elif _polarization is True:
            _r_p, _t_p = _fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            return jnp.array([_r_p, _t_p])

def _tmm(stack: Stack, light: Light,
         polarization: Optional[Union[str, bool]],
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
    def calculate_rt(stack, polarization, theta, wavelength):
        """
        This helper function computes the reflection and transmission coefficients, denoted as R and T, for the multilayer 
        thin film. The calculation is performed using the Transfer Matrix Method (TMM), which involves constructing 
        characteristic matrices for each layer, considering the angle of incidence, wavelength, and polarization of the light.
        """
        # Core logic for TMM that takes into account coherent or incoherent layers
        # Here, you would implement the necessary equations and matrix multiplications for R and T.
        # Depending on whether the layers are coherent or incoherent, different approaches will be applied.
        
        

        return R, T

    # Helper function to calculate the reflection (r) and transmission (t) coefficients for coherent layers
    def calculate_rt_coherent(stack, polarization, theta, wavelength):
        """
        This helper function computes the reflection (r) and transmission (t) coefficients, which are applicable for coherent
        layers in the multilayer thin film. The r and t coefficients are complex quantities, from which reflectance, 
        transmittance, and other derived quantities like absorbed energy and ellipsometric data are computed.
        """
        
        layer_phases = stack.kz * stack.thicknesses
        
        
        
        return r, t

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

    # Main computation
    theta = light.angle_of_incidence[theta_index]
    wavelength = light.wavelength[wavelength_index]
    if stack.any_incoherent:
        # Calculate reflectance (R) and transmittance (T) using incoherent layer assumptions
        R, T = calculate_rt(stack, polarization, theta, wavelength)
        return R, T
    else:
        # Calculate reflection (r) and transmission (t) using coherent layer assumptions
        r, t = calculate_rt_coherent(stack, polarization, theta, wavelength)

        # Reflectance and transmittance are derived from r and t for coherent layers
        R = jnp.abs(r)**2
        T = jnp.abs(t)**2

        # Conditional outputs based on Stack object flags
        results = [R, T]

        if stack.obs_absorbed_energy:
            absorbed_energy = calculate_absorbed_energy(stack, r, t)
            results.append(absorbed_energy)

        if stack.obs_ellipsiometric:
            psi, delta = calculate_ellipsometric_data(stack, r, t)
            results.extend([psi, delta])

        if stack.obs_poynting:
            poynting_vector = calculate_poynting_vector(stack, r, t)
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
