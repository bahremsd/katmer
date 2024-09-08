import jax.numpy as jnp
from jax import vmap
import jax
import numpy as np
import sys
from typing import Optional, Union, Tuple, Dict, Callable, List

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
        elif polarization == 's' or polarization == False:
            self._polarization = False  # s-polarization corresponds to False
        elif polarization == 'p' or polarization == True:
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

    def save_log(self, wavelength_file: str, angle_file: str, polarization_file: str):
        """
        Save wavelength, angle of incidence, and polarization data to CSV files.

        Parameters:
        - wavelength_file: str
            Path to the file (without extension) where wavelength data will be saved.
        - angle_file: str
            Path to the file (without extension) where angle of incidence data will be saved.
        - polarization_file: str
            Path to the file (without extension) where polarization state will be saved.
        """
        # Append .csv to filenames
        wavelength_file = wavelength_file + '.csv'
        angle_file = angle_file + '.csv'
        polarization_file = polarization_file + '.csv'
        
        # Convert wavelength to numpy array and save to CSV
        np.savetxt(wavelength_file, self._wavelength, header='wavelength', delimiter=',', comments='')

        # Convert angle of incidence to numpy array and save to CSV
        np.savetxt(angle_file, self._angle_of_incidence, header='angle_of_incidence', delimiter=',', comments='')

        # Handle polarization data
        if self._polarization is not None:
            # Convert boolean to string ('s' or 'p')
            polarization_value = 'p' if self._polarization else 's'
        else:
            # Set value to "unpolarized"
            polarization_value = 'unpolarized'
        
        # Save polarization data to CSV
        with open(polarization_file, 'w') as file:
            file.write('polarization\n')  # Write header
            file.write(f'{polarization_value}\n')  # Write data



def is_propagating_wave(n: Union[float, jnp.ndarray], theta: Union[float, jnp.ndarray], polarization: Optional[bool] = None) -> Union[bool, Tuple[bool, bool]]:
    """
    Determines whether the wave is propagating forward through the stack based on the angle, 
    refractive index, and polarization.
    
    Args:
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




    def define_is_forward_if_bigger_than_eps(_):
        # For evanescent or lossy mediums, forward is determined by decay
        is_forward_s = jnp.sign(n_cos_theta.imag)
        is_forward_p = is_forward_s  # The decay condition applies to both polarizations equally
        return is_forward_s, is_forward_p
    
    def define_is_forward_if_smaller_than_eps(_):
        # For s-polarization: Re[n cos(theta)] > 0
        is_forward_s = jnp.sign(n_cos_theta.real)
        
        # For p-polarization: Re[n cos(theta*)] > 0
        n_cos_theta_star = n * jnp.cos(jnp.conj(theta))
        is_forward_p = jnp.sign(n_cos_theta_star.real)
        return is_forward_s, is_forward_p

    # Handle the evanescent and lossy cases by checking the imaginary part
    condition = jnp.abs(n_cos_theta.imag) > EPSILON * 1e3
    is_forward_s, is_forward_p = jax.lax.cond(condition, define_is_forward_if_bigger_than_eps, define_is_forward_if_smaller_than_eps, None)
    # Return based on polarization argument
    if polarization is None:
        # Unpolarized case: Return tuple (s-polarization, p-polarization)
        return jnp.array([is_forward_s, is_forward_p])
    elif polarization is False:
        # s-polarization case
        return jnp.array([is_forward_s])
    elif polarization is True:
        # p-polarization case
        return jnp.array([is_forward_p])


def _compute_layer_angles_one_theta_one_wl(nk_functions: Dict[int, Callable],
                                    material_distribution: List[int], 
                                    initial_theta: Union[float, jnp.ndarray], 
                                    wavelength: Union[float, jnp.ndarray],
                                    polarization: Optional[bool],
                                    incoming_medium: Callable,
                                    outgoing_medium: Callable) -> jnp.ndarray:
    """
    Computes the angle of incidence for light in each layer of a multilayer thin film using Snell's law 
    (just for 1 wl and init theta nk value).

    Args:
        nk_functions (Dict[int, Callable]): A dictionary where each key corresponds to an index 
                                            in the material_distribution, and each value is a function 
                                            that takes wavelength as input and returns the complex 
                                            refractive index for that material.

        material_distribution (List[int]): A list that describes the distribution of materials across 
                                           the layers. Each element is an index to the nk_functions dictionary.  

        initial_theta (Union[float, Array]): The angle of incidence (in radians) with respect to 
                                             the normal of the first layer. This argument can either 
                                             be a single float value (for single angle processing) 
                                             or a one-dimensional JAX array (for batch processing).

        wavelength (Union[float, jnp.ndarray]): The wavelength or an array of wavelengths (ndarray) 
                                               for which the computation will be performed.

        polarization (Optional[bool]): Determines the polarization state:
            - None: Unpolarized (returns a tuple of booleans for both s and p polarizations).
            - False: s-polarization (returns a boolean for s-polarization).
            - True: p-polarization (returns a boolean for p-polarization).     

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
    # Create a function that retrieves the refractive indices for each material in the distribution
    def get_nk_values(wl):
        # For each material in the distribution, call the corresponding nk function with the given wavelength
        return jnp.array([nk_functions[mat_idx](wl) for mat_idx in material_distribution])
    
    nk_list = get_nk_values(wavelength)

    incoming_medium_nk = jnp.expand_dims(incoming_medium(wavelength), axis=0)
    outgoing_medium_nk = jnp.expand_dims(outgoing_medium(wavelength), axis=0)

    concatenated_nk_list = jnp.concatenate([incoming_medium_nk, nk_list, outgoing_medium_nk], axis=0)

    # Calculate the sine of the angles in the first layer using Snell's law
    sin_theta = jnp.sin(initial_theta) * concatenated_nk_list[0] / concatenated_nk_list
    # Compute the angle (theta) in each layer using the arcsin function
    # jnp.arcsin is preferred for compatibility with complex values if needed
    theta_array = jnp.arcsin(sin_theta)
    # If the angle is not forward-facing, we subtract it from pi to flip the orientation.
    is_incoming_props = is_propagating_wave(concatenated_nk_list[0], theta_array[0], polarization)
    is_outgoing_props = is_propagating_wave(concatenated_nk_list[-1], theta_array[-1], polarization)

    def update_theta_arr_incoming(_):
        return theta_array.at[0].set(jnp.pi - theta_array[0])

    def update_theta_arr_outgoing(_):
        return theta_array.at[-1].set(jnp.pi - theta_array[-1])
    
    def return_unchanged_theta(_):
        return theta_array

    # Handle the evanescent and lossy cases by checking the imaginary part
    condition_incoming = jnp.any(jnp.logical_or(jnp.all(is_incoming_props == jnp.array([1, 1])), is_incoming_props == jnp.array([1])))
    condition_outgoing = jnp.any(jnp.logical_or(jnp.all(is_outgoing_props == jnp.array([1, 1])), is_outgoing_props == jnp.array([1])))

    theta_array = jax.lax.cond(condition_incoming, update_theta_arr_incoming, return_unchanged_theta, operand=None)
    theta_array = jax.lax.cond(condition_outgoing, update_theta_arr_outgoing, return_unchanged_theta, operand=None)

    # Return a 1D theta array for each layer
    return theta_array

def compute_layer_angles(nk_functions: Dict[int, Callable], 
                         material_distribution: List[int], 
                         initial_theta: Union[float, jnp.ndarray], 
                         wavelength: Union[float, jnp.ndarray],
                         polarization: Optional[bool],
                         incoming_medium: Callable,
                         outgoing_medium: Callable) -> jnp.ndarray:
    """
    Calculates the angles of incidence across layers for a set of refractive indices (nk_list_2d) 
    and an initial angle of incidence (initial_theta) using vectorization.

    Args:
        nk_functions (Dict[int, Callable]): A dictionary where each key corresponds to an index 
                                            in the material_distribution, and each value is a function 
                                            that takes wavelength as input and returns the complex 
                                            refractive index for that material.

        material_distribution (List[int]): A list that describes the distribution of materials across 
                                           the layers. Each element is an index to the nk_functions dictionary.

        initial_theta (Union[float, jnp.ndarray]): The initial angle of incidence (in radians). Can be 
                                                  a single float or a 1D/2D jax array (ndarray) depending 
                                                  on the use case.

        wavelength (Union[float, jnp.ndarray]): The wavelength or an array of wavelengths (ndarray) 
                                               for which the computation will be performed.

        polarization (Optional[bool]): Determines the polarization state:
            - None: Unpolarized (returns a tuple of booleans for both s and p polarizations).
            - False: s-polarization (returns a boolean for s-polarization).
            - True: p-polarization (returns a boolean for p-polarization).  

    Returns:
        jnp.ndarray: A 3D JAX array where the [i, j, :] entry represents the angles of incidence 
                     for the j-th initial angle at the i-th wavelength. The size of the third dimension 
                     corresponds to the number of layers.
    """
    initial_theta = jnp.array(initial_theta, dtype = float)
    wavelength = jnp.array(wavelength, dtype = float)
    # Vectorize the _compute_layer_angles_one_theta_one_wl function over the wavelength dimension (first dimension of nk_list_2d)
    # in_axes=(0, None, 0) means:
    # - The first argument (nk_list_2d) will not be vectorized
    # - The second argument (initial_theta) will be vectorized over the first dimension
    vmap_compute_layer_angles = vmap(vmap(_compute_layer_angles_one_theta_one_wl, (None,None, None, 0, None, None, None)), (None,None, 0, None, None, None, None))

    # Apply the vectorized function to get the 3D array of angles
    # The resulting array has dimensions (number_of_wavelengths, number_of_init_angles, number_of_layers)
    return vmap_compute_layer_angles(nk_functions, material_distribution, initial_theta, wavelength, polarization, incoming_medium, outgoing_medium)