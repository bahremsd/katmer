from typing import Optional, Union, Tuple
import jax.numpy as jnp
from jax import vmap, jit

from stacks import Stack
from light import Light

def _interface(
    are_boundary_have_incoherency: bool,
    polarization: Optional[bool],
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
    are_boundary_have_incoherency : bool
        Flag indicating if any of the layers at the boundary are incoherent.
        - False: Both boundaries are coherent.
        - True: At least one boundary is incoherent.
    
    polarization : Optional[bool]
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

    # Fresnel equations for s-polarization (electric field perpendicular to the plane of incidence)
    def _fresnel_s(_first_layer_n: Union[float, jnp.ndarray], _second_layer_n: Union[float, jnp.ndarray],
                   _first_layer_theta: Union[float, jnp.ndarray], _second_layer_theta: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        _r_s = (_first_layer_n * jnp.cos(_first_layer_theta) - _second_layer_n * jnp.cos(_second_layer_theta)) / \
                (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta))
        _t_s = 2 * _first_layer_n * jnp.cos(_first_layer_theta) / \
                (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta))
        return _r_s, _t_s

    # Fresnel equations for p-polarization (electric field parallel to the plane of incidence)
    def _fresnel_p(_first_layer_n: Union[float, jnp.ndarray], _second_layer_n: Union[float, jnp.ndarray],
                   _first_layer_theta: Union[float, jnp.ndarray], _second_layer_theta: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        _r_p = (_second_layer_n * jnp.cos(_first_layer_theta) - _first_layer_n * jnp.cos(_second_layer_theta)) / \
                (_second_layer_n * jnp.cos(_first_layer_theta) + _first_layer_n * jnp.cos(_second_layer_theta))
        _t_p = 2 * _first_layer_n * jnp.cos(_first_layer_theta) / \
                (_second_layer_n * jnp.cos(_first_layer_theta) + _first_layer_n * jnp.cos(_second_layer_theta))
        return _r_p, _t_p

    # Handle the incoherent case
    if are_boundary_have_incoherency:
        if polarization is None:
            # Unpolarized light: both s and p polarizations
            _r_s, _t_s = _fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            _r_p, _t_p = _fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            _R_s = jnp.abs(_r_s) ** 2
            _T_s = 1 - _R_s
            _R_p = jnp.abs(_r_p) ** 2
            _T_p = 1 - _R_p
            return jnp.array([[_R_s, _R_p], [_T_s, _T_p]])
        elif polarization is False:
            _r_s, _t_s = _fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            _R_s = jnp.abs(_r_s) ** 2
            _T_s = 1 - _R_s
            return jnp.array([_R_s, _T_s])
        elif polarization is True:
            _r_p, _t_p = _fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            _R_p = jnp.abs(_r_p) ** 2
            _T_p = 1 - _R_p
            return jnp.array([_R_p, _T_p])

    # Handle the coherent case
    else:
        if polarization is None:
            # Unpolarized light: both s and p polarizations
            _r_s, _t_s = _fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            _r_p, _t_p = _fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            return jnp.array([[_r_s, _r_p], [_t_s, _t_p]])
        elif polarization is False:
            _r_s, _t_s = _fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            return jnp.array([_r_s, _t_s])
        elif polarization is True:
            _r_p, _t_p = _fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
            return jnp.array([_r_p, _t_p])

def _tmm(stack: Stack,
         polarization: Optional[Union[str, bool]],
         theta: Union[float, jnp.ndarray],
         wavelength: Union[float, jnp.ndarray]
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
        theta (Union[float, jnp.ndarray]): The angle of incidence (in radians) at which light strikes the multilayer thin 
                                           film. This can be a single float value or a jnp array for angle-dependent 
                                           computations.
        wavelength (Union[float, jnp.ndarray]): The wavelength(s) of the incident light. This can be a single float value 
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
        
        if stack.are_there_any_incoherent_layer:
            # Perform computations assuming incoherent layers
            # ...
            pass
        else:
            # Perform computations assuming coherent layers
            # ...
            pass

        return R, T

    # Helper function to calculate the reflection (r) and transmission (t) coefficients for coherent layers
    def calculate_rt_coherent(stack, polarization, theta, wavelength):
        """
        This helper function computes the reflection (r) and transmission (t) coefficients, which are applicable for coherent
        layers in the multilayer thin film. The r and t coefficients are complex quantities, from which reflectance, 
        transmittance, and other derived quantities like absorbed energy and ellipsometric data are computed.
        """
        # Perform the calculations for coherent layers, resulting in complex coefficients r and t
        # ...
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

    # Main computation logic
    if stack.are_there_any_incoherent_layer:
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
def forward(_stack: Stack, _light: Light) -> Union[
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
    _polarization = _light.polarization  # 's', 'p', or None (unpolarized)
    _theta = _light.theta  # Array or single value for angle of incidence
    _wavelength = _light.wavelength  # Array or single value for wavelength

    # Vectorize the _tmm function across theta and wavelength using JAX's vmap
    _tmm_vectorized = vmap(
        _tmm, 
        in_axes=(None, None, 0, 0)  # Fix _stack and _polarization, vmap _theta and _wavelength
    )

    # Apply the vectorized function to the theta and wavelength arrays
    _result = _tmm_vectorized(_stack, _polarization, _theta, _wavelength)

    # Return the result
    return _result
