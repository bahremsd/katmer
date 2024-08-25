import jax.numpy as jnp
import sys
from typing import Optional, Union, Tuple, List

# Define EPSILON as the smallest representable positive number such that 1.0 + EPSILON != 1.0
EPSILON = sys.float_info.epsilon

class Light:
    def __init__(self, 
                 wavelength: jnp.ndarray, 
                 angle_of_incidence: jnp.ndarray, 
                 polarization: Optional[Union[str, bool]] = None):
        """
        Initialize the Light class.
        
        This class serves as a data container for encapsulating the properties 
        of light waves that are used in optical simulations in multilayer thin 
        films, particularly in the context of the Transfer Matrix Method (TMM). 
        The `Light` class stores essential attributes such as the wavelength of 
        the light, the angle of incidence, and the polarization state. It is 
        designed to ensure that these attributes are correctly formatted and typed, 
        specifically leveraging JAX arrays for high-performance numerical 
        computations.

        Parameters:
        - wavelength: jax.numpy.ndarray
            Array of wavelengths (in micrometers or nanometers).
        - angle_of_incidence: jax.numpy.ndarray
            Array of angles of incidence (in degrees).
        - polarization: Optional[Union[str, bool]]
            Polarization state: 's' for s-polarization, 'p' for p-polarization, 
            or None for unpolarized light.

        Raises:
        - TypeError:
            If `wavelength` or `angle_of_incidence` are not instances of `jax.numpy.ndarray`.
        - ValueError:
            If `polarization` is not 's', 'p', or None.
        """
        if not isinstance(wavelength, jnp.ndarray):
            raise TypeError("The wavelength array must be of type jax.numpy.ndarray.")
        
        if not isinstance(angle_of_incidence, jnp.ndarray):
            raise TypeError("The angle_of_incidence array must be of type jax.numpy.ndarray.")

        self._wavelength = jnp.array(wavelength)
        self._angle_of_incidence = jnp.array(angle_of_incidence)

        if polarization is None:
            self._polarization = None
        elif polarization == 's':
            self._polarization = False  # s-polarization corresponds to False
        elif polarization == 'p':
            self._polarization = True   # p-polarization corresponds to True
        else:
            raise ValueError("Invalid value for polarization. "
                             "It must be 's', 'p', or None for unpolarized light.")

    @property
    def wavelength(self) -> jnp.ndarray:
        """
        Getter for the wavelength array.

        Returns:
        - jax.numpy.ndarray
            Array of wavelengths.
        """
        return self._wavelength

    @property
    def angle_of_incidence(self) -> jnp.ndarray:
        """
        Getter for the angle of incidence array.

        Returns:
        - jax.numpy.ndarray
            Array of angles of incidence.
        """
        return self._angle_of_incidence

    @property
    def polarization(self) -> Optional[bool]:
        """
        Getter for the polarization state.

        Returns:
        - Optional[bool]
            Polarization state: 
            - False for s-polarization
            - True for p-polarization
            - None for unpolarized light
        """
        return self._polarization

def is_propagating_wave(n: Union[float, jnp.ndarray], theta: Union[float, jnp.ndarray], polarization: Optional[bool] = None) -> Union[bool, Tuple[bool, bool]]:
    """
    Determines whether the wave is propagating forward through the stack based on the angle, 
    refractive index, and polarization.
    
    Parameters:
    n (Union[float, jnp.ndarray]): Refractive index of the medium (can be complex).
    theta (Union[float, jnp.ndarray]): Angle of incidence (in radians) with respect to the normal.
    polarization (Optional[bool]): Determines the polarization state:
        - None: Unpolarized (returns a tuple of booleans for both s and p polarizations).
        - False: s-polarization (returns a boolean for s-polarization).
        - True: p-polarization (returns a boolean for p-polarization).
    
    Returns:
    Union[bool, Tuple[bool, bool]]:
        - If polarization is None, returns a tuple of booleans (s-polarization, p-polarization).
        - If polarization is False, returns a boolean indicating if the wave is forward-propagating for s-polarization.
        - If polarization is True, returns a boolean indicating if the wave is forward-propagating for p-polarization.
    
    The function evaluates whether the wave, given its angle, refractive index, and polarization, is a 
    forward-propagating wave (i.e., traveling from the front to the back of the stack). This is crucial 
    when calculating Snell's law in multilayer structures to ensure light is correctly entering or 
    exiting the stack.
    
    The check considers both real and complex values of the refractive index and angle, ensuring that the 
    light propagates within the correct angle range for physical interpretation.
    """
    # Calculate n * cos(theta) to evaluate propagation direction for s-polarization
    n_cos_theta = n * jnp.cos(theta)
    
    # Handle the evanescent and lossy cases by checking the imaginary part
    if jnp.abs(n_cos_theta.imag) > EPSILON * 1e3:
        # For evanescent or lossy mediums, forward is determined by decay
        is_forward_s = n_cos_theta.imag > 0
        is_forward_p = is_forward_s  # The decay condition applies to both polarizations equally
    else:
        # For s-polarization: Re[n cos(theta)] > 0
        is_forward_s = n_cos_theta.real > 0
        
        # For p-polarization: Re[n cos(theta*)] > 0
        n_cos_theta_star = n * jnp.cos(jnp.conj(theta))
        is_forward_p = n_cos_theta_star.real > 0

    # Return based on polarization argument
    if polarization is None:
        # Unpolarized case: Return tuple (s-polarization, p-polarization)
        return bool(is_forward_s), bool(is_forward_p)
    elif polarization is False:
        # s-polarization case
        return bool(is_forward_s)
    elif polarization is True:
        # p-polarization case
        return bool(is_forward_p)


def compute_layer_angles(n_list: jnp.ndarray, initial_theta: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """
    Computes the angle of incidence for light in each layer of a multilayer thin film using Snell's law.

    Args:
        n_list (Array): A one-dimensional JAX array representing the complex refractive indices 
                        of the materials in each layer of the multilayer thin film. Each element 
                        is of the form `n + j*k`, where `n` is the refractive index, and `k` 
                        is the extinction coefficient, which accounts for the absorption in the material.
                        
        initial_theta (Union[float, Array]): The angle of incidence (in radians) with respect to 
                                             the normal of the first layer. This argument can either 
                                             be a single float value (for single angle processing) 
                                             or a one-dimensional JAX array (for batch processing).
                                             
    Returns:
        Array: A JAX array containing the calculated angles of incidence for each layer.
               - If `initial_theta` is a float or 0-D, the function returns a one-dimensional array where 
                 each element represents the angle of incidence in a specific layer.
               - If `initial_theta` is a one-dimensional array, the function returns a two-dimensional 
                 array. Each row of this array corresponds to the angles of incidence across all layers 
                 for a specific initial angle provided in `initial_theta`.
    
    Detailed Description:
        The function starts by ensuring that the `initial_theta` input is treated as a one-dimensional 
        array. Next, the function applies Snell's law to calculate the sine of the angle in each layer. 
        Snell's law relates the angle of incidence and the refractive indices of two media through 
        the equation: 
            sin(theta_i) = (n_0 * sin(theta_0)) / n_i
        where:
            - theta_i is the angle of incidence in the ith layer,
            - n_0 is the refractive index of the first layer,
            - theta_0 is the initial angle of incidence,
            - n_i is the refractive index of the ith layer.    
    """
    # Ensure that the initial angle input is treated as a one-dimensional array.
    # This is crucial for consistent processing of both single and batch angle inputs.
    initial_theta = jnp.atleast_1d(initial_theta)
    
    # Calculate the sine of the angles in the first layer using Snell's law
    sin_theta = jnp.sin(initial_theta) * n_list[0] / n_list
    
    # Compute the angle (theta) in each layer using the arcsin function
    # jnp.arcsin is preferred for compatibility with complex values if needed
    theta_array = jnp.arcsin(sin_theta)
    # If the angle is not forward-facing, we subtract it from pi to flip the orientation.
    if not is_propagating_wave(n_list[0], theta_array[0]):
        theta_array = theta_array.at[0].set(jnp.pi - theta_array[0])
    if not is_propagating_wave(n_list[-1], theta_array[-1]):
        theta_array = theta_array.at[-1].set(jnp.pi - theta_array[-1])
        
    # If only one initial theta is provided, return a 1D array; otherwise, return a 2D array
    return theta_array if initial_theta.ndim > 1 else theta_array.squeeze(axis=0)
