import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
from typing import Callable, List, Dict, Union, Tuple


from katmer.data import interpolate_nk
from katmer.light import compute_layer_angles


def _fresnel_s(_first_layer_n: Union[float, jnp.ndarray], 
               _second_layer_n: Union[float, jnp.ndarray],
               _first_layer_theta: Union[float, jnp.ndarray], 
               _second_layer_theta: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This function calculates the Fresnel reflection (r_s) and transmission (t_s) coefficients 
    for s-polarized light (electric field perpendicular to the plane of incidence) at the interface 
    between two materials. The inputs are the refractive indices and the angles of incidence and 
    refraction for the two layers.

    Args:
        _first_layer_n (Union[float, jnp.ndarray]): Refractive index of the first layer (incident medium). 
            Can be a float or an array if computing for multiple incident angles/materials.
        _second_layer_n (Union[float, jnp.ndarray]): Refractive index of the second layer (transmitted medium). 
            Similar to the first argument, this can also be a float or an array.
        _first_layer_theta (Union[float, jnp.ndarray]): Angle of incidence in the first layer (in radians). 
            Can be a float or an array.
        _second_layer_theta (Union[float, jnp.ndarray]): Angle of refraction in the second layer (in radians). 
            Can be a float or an array.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing two jax.numpy arrays:
            - r_s: The Fresnel reflection coefficient for s-polarized light.
            - t_s: The Fresnel transmission coefficient for s-polarized light.
    """
    
    # Calculate the reflection coefficient for s-polarized light using Fresnel's equations.
    # The formula: r_s = (n1 * cos(theta1) - n2 * cos(theta2)) / (n1 * cos(theta1) + n2 * cos(theta2))
    # This measures how much of the light is reflected at the interface.
    r_s = ((_first_layer_n * jnp.cos(_first_layer_theta) - _second_layer_n * jnp.cos(_second_layer_theta)) /
           (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))
    
    # Calculate the transmission coefficient for s-polarized light using Fresnel's equations.
    # The formula: t_s = 2 * n1 * cos(theta1) / (n1 * cos(theta1) + n2 * cos(theta2))
    # This measures how much of the light is transmitted through the interface.
    t_s = (2 * _first_layer_n * jnp.cos(_first_layer_theta) /
           (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))
    
    # Return the reflection and transmission coefficients as a JAX array
    return jnp.array([r_s, t_s])


def _fresnel_p(_first_layer_n: Union[float, jnp.ndarray], 
               _second_layer_n: Union[float, jnp.ndarray],
               _first_layer_theta: Union[float, jnp.ndarray], 
               _second_layer_theta: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This function calculates the Fresnel reflection (r_p) and transmission (t_p) coefficients 
    for p-polarized light at the interface between two different media. It uses the refractive indices
    of the two media (_first_layer_n and _second_layer_n) and the incident and transmitted angles
    (_first_layer_theta and _second_layer_theta) to compute these values.

    Args:
        _first_layer_n: Refractive index of the first medium (can be float or ndarray).
        _second_layer_n: Refractive index of the second medium (can be float or ndarray).
        _first_layer_theta: Incident angle (in radians) in the first medium (can be float or ndarray).
        _second_layer_theta: Transmitted angle (in radians) in the second medium (can be float or ndarray).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing two arrays:
            - r_p: The reflection coefficient for p-polarized light.
            - t_p: The transmission coefficient for p-polarized light.
    """

    # Calculate the reflection coefficient for p-polarized light (r_p)
    # This equation is based on the Fresnel equations for p-polarization, where 
    # r_p is the ratio of the reflected and incident electric field amplitudes for p-polarized light.
    r_p = ((_second_layer_n * jnp.cos(_first_layer_theta) - _first_layer_n * jnp.cos(_second_layer_theta)) /
           (_second_layer_n * jnp.cos(_first_layer_theta) + _first_layer_n * jnp.cos(_second_layer_theta)))

    # Calculate the transmission coefficient for p-polarized light (t_p)
    # This equation is also derived from the Fresnel equations for p-polarization.
    # t_p represents the ratio of the transmitted and incident electric field amplitudes.
    t_p = (2 * _first_layer_n * jnp.cos(_first_layer_theta) /
           (_second_layer_n * jnp.cos(_first_layer_theta) + _first_layer_n * jnp.cos(_second_layer_theta)))

    # Return the reflection and transmission coefficients as a tuple of jnp arrays
    # Both r_p and t_p are essential for understanding how light interacts with different layers.
    return jnp.array([r_p, t_p])


def _compute_rt_at_interface_s(carry, concatenated_nk_list_theta):
    """
    This function calculates the reflection (r) and transmission (t) coefficients 
    for s-polarization at the interface between two layers in a multilayer thin-film system. 
    It uses the Fresnel equations for s-polarized light. The function is designed to be used 
    in a JAX `lax.scan` loop, where it processes each interface iteratively.
    
    Args:
        carry: A tuple containing the index (carry_idx) and a matrix (carry_values) 
               where the reflection and transmission coefficients will be stored.
               - carry_idx (int): The current index, indicating which layer interface is being processed.
               - carry_values (array): An array to store the r,t coefficients for each interface.
        
        concatenated_nk_list_theta: A tuple containing two arrays:
               - stacked_nk_list (array): The refractive indices (n) of two consecutive layers at the interface.
               - stacked_layer_angles (array): The angles of incidence for the two consecutive layers.

    Returns:
        A tuple of:
            - Updated carry: The new index and updated matrix with the calculated r,t coefficients.
            - None: Required to match the JAX `lax.scan` interface, where a second argument is expected.
    """

    # Unpack the concatenated list into refractive index list and angle list
    stacked_nk_list, stacked_layer_angles = concatenated_nk_list_theta
    # `stacked_nk_list`: contains the refractive indices of two consecutive layers at the interface
    # `stacked_layer_angles`: contains the angles of incidence for these two layers

    # Unpack the carry tuple
    carry_idx, carry_values = carry
    # `carry_idx`: current index in the process, starts from 0 and iterates over layer interfaces
    # `carry_values`: the array that stores the reflection and transmission coefficients

    # Compute the reflection and transmission coefficients using the Fresnel equations for s-polarization
    r_t_matrix = _fresnel_s(_first_layer_theta=stacked_layer_angles[0],   # Incident angle of the first layer
                            _second_layer_theta=stacked_layer_angles[1],  # Incident angle of the second layer
                            _first_layer_n=stacked_nk_list[0],            # Refractive index of the first layer
                            _second_layer_n=stacked_nk_list[1])           # Refractive index of the second layer
    # This line computes r and t coefficients between two consecutive layers 
    # based on their refractive indices and angles of incidence.

    # Store the computed r,t matrix in the `carry_values` array at the current index
    carry_values = carry_values.at[carry_idx, :].set(r_t_matrix)  # Set r,t coefficients at the current index
    # The `carry_values.at[carry_idx, :].set(r_t_matrix)` updates the array at position `carry_idx` 
    # with the computed r,t coefficients.

    carry_idx = carry_idx + 1  # Move to the next index for the next iteration
    # The carry index is incremented to process the next layer interface in subsequent iterations.

    # Return the updated carry (with new index and r,t coefficients) and None for lax.scan compatibility
    return (carry_idx, carry_values), None


def _compute_rt_at_interface_p(carry, concatenated_nk_list_theta):
    """
    This function computes the reflection and transmission (r, t) coefficients at the interface between two layers
    for P-polarized light (parallel polarization). It uses the Fresnel equations to calculate these coefficients 
    based on the refractive indices and angles of incidence and refraction for the two layers.

    Args:
        carry: A tuple (carry_idx, carry_values) where:
            - carry_idx: The current index that keeps track of the layer.
            - carry_values: A matrix to store the computed reflection and transmission coefficients.
        
        concatenated_nk_list_theta: A tuple (stacked_nk_list, stacked_layer_angles) where:
            - stacked_nk_list: A list of refractive indices of the two consecutive layers.
            - stacked_layer_angles: A list of angles of incidence and refraction at the interface between the layers.

    Returns:
        A tuple:
            - Updated carry containing:
                - carry_idx incremented by 1.
                - carry_values with the newly computed r, t coefficients at the current interface.
            - None (This is used to maintain the structure of a functional-style loop but has no further use).
    """

    # Unpack the concatenated data into two variables: refractive indices (nk) and angles (theta)
    stacked_nk_list, stacked_layer_angles = concatenated_nk_list_theta  # Extract the refractive indices and angles from the input tuple
    carry_idx, carry_values = carry  # Unpack carry: carry_idx is the current index, carry_values stores r and t coefficients

    # Compute reflection (r) and transmission (t) coefficients at the interface using Fresnel equations for P-polarized light
    r_t_matrix = _fresnel_p(_first_layer_theta = stacked_layer_angles[0],  # Incident angle at the first layer
                              _second_layer_theta = stacked_layer_angles[1],  # Refraction angle at the second layer
                              _first_layer_n = stacked_nk_list[0],  # Refractive index of the first layer
                              _second_layer_n = stacked_nk_list[1])  # Refractive index of the second layer

    # Update carry_values by setting the r,t matrix at the current index (carry_idx)
    carry_values = carry_values.at[carry_idx, :].set(r_t_matrix)  # Store the computed r,t matrix into the carry_values at the index 'carry_idx'

    carry_idx = carry_idx + 1  # Move to the next index for further iterations
    return (carry_idx, carry_values), None  # Return the updated carry with incremented index and updated r,t values, and None as a placeholder

def _compute_rt_one_wl(nk_list: jnp.ndarray, layer_angles: jnp.ndarray,
                       wavelength: Union[float, jnp.ndarray], polarization: bool) -> jnp.ndarray:
    """
    Computes the reflectance and transmittance for a single wavelength 
    across multiple layers in a stack of materials. The computation 
    takes into account the refractive index of each layer, the angle of 
    incidence in each layer, the wavelength of the light, and the 
    polarization of the light.

    Args:
        nk_list (jnp.ndarray): Array of complex refractive indices for each layer. 
                               The shape should be (num_layers,).
        layer_angles (jnp.ndarray): Array of angles of incidence for each layer. 
                                    The shape should be (num_layers,).
        wavelength (float or jnp.ndarray): The wavelength of light, given as either 
                                           a scalar or a JAX array.
        polarization (bool): Boolean flag that determines the polarization state of the light. 
                             If False, s-polarization is used; if True, p-polarization is used.

    Returns:
        jnp.ndarray: A 1D JAX array representing the reflectance and transmittance 
                     coefficients at the specified wavelength and polarization.
    """

    # Initialize the state for `jax.lax.scan`. The first element (0) is a placeholder 
    # and won't be used. The second element is a 2D array of zeros to hold intermediate 
    # results, representing the reflectance and transmittance across layers.
    init_state = (0, jnp.zeros((len(nk_list) - 2, 2), dtype=jnp.float32))  # Initial state with an array of zeros
    # The shape of `jnp.zeros` is (num_layers - 2, 2) because we exclude the first 
    # and last layers, assuming they are boundary layers.

    # Stack the refractive indices (`nk_list`) for each adjacent pair of layers.
    # This creates a new array where each element contains a pair of adjacent refractive indices 
    # from `nk_list`, which will be used to compute the reflection and transmission at the interface 
    # between these two layers.
    stacked_nk_list = jnp.stack([nk_list[:-2], nk_list[1:-1]], axis=1)  # Stack the original and shifted inputs for processing in pairs
    # For example, if `nk_list` is [n1, n2, n3, n4], this will create pairs [(n1, n2), (n2, n3), (n3, n4)].

    # Similarly, stack the angles for adjacent layers.
    # The same logic applies to `layer_angles` as for `nk_list`. Each pair of adjacent layers 
    # will have an associated pair of angles.
    stacked_layer_angles = jnp.stack([layer_angles[:-2], layer_angles[1:-1]], axis=1)
    # This operation aligns the angles with the corresponding refractive indices.

    # Now we need to compute reflectance and transmittance for each interface. 
    # This can be done using `jax.lax.scan`, which efficiently loops over the stacked pairs 
    # of refractive indices and angles.

    # If the light is s-polarized (polarization = False), we call the function `_compute_rt_at_interface_s`.
    # This function calculates the reflection and transmission coefficients specifically for s-polarized light.
    if polarization == False:
        rt_one_wl, _ = jax.lax.scan(_compute_rt_at_interface_s, init_state, (stacked_nk_list, stacked_layer_angles))  # s-polarization case
        # `jax.lax.scan` applies the function `_compute_rt_at_interface_s` to each pair of adjacent layers 
        # along with the corresponding angles. It processes this in a loop, accumulating the results.

    # If the light is p-polarized (polarization = True), we use `_compute_rt_at_interface_p` instead.
    # This function handles p-polarized light.
    elif polarization == True:
        rt_one_wl, _ = jax.lax.scan(_compute_rt_at_interface_p, init_state, (stacked_nk_list, stacked_layer_angles))  # p-polarization case
        # The same process as above but with a function specific to p-polarized light.

    # Finally, return the computed reflectance and transmittance coefficients. 
    # The result is stored in `rt_one_wl[1]` (the second element of `rt_one_wl`), which corresponds 
    # to the reflectance and transmittance after all layers have been processed.
    return rt_one_wl[1]  # Return a 1D theta array for each layer
    # This output is the desired result: the reflectance and transmittance for the given wavelength.


class Stack:
    """
    Stack class to simulate multilayer thin films using the Transfer Matrix Method (TMM).
    This class is designed for inverse design simulations and includes advanced features for
    handling material properties, layer thicknesses, and incoherency options.
    """

    def __init__(self, auto_coherency: bool = True, any_incoherent: bool = False,
                 fixed_material_distribution: bool = False, incoming_medium: str = "Air",
                 outgoing_medium: str = "Air", obs_absorbed_energy: bool = False, obs_ellipsiometric: bool = False,
                 obs_poynting: bool = False, *args, **kwargs):
        """
        Initialize the Stack class with material data, layer thicknesses, and coherency options.
        
        Args:
        - auto_coherency (bool): Flag to determine whether coherency should be automatically handled.
        - any_incoherent (bool): Flag to indicate whether there are any incoherent layers in the stack.
        - fixed_material_distribution (bool): Determines if material distribution is fixed.
        - incoming_medium (str): This represents the nk information for the medium 
                      through which light enters the stack. Although this medium can be considered as a layer
                      with infinite thickness. Default is air.
        - outgoing_medium (str): This represents the nk information for the medium 
                      through which light exits the stack. Although this medium can be considered as a layer
                      with infinite thickness. Default is air.
        - obs_absorbed_energy (bool): Flag to determine whether the absorbed energy in the stack should be observed
                                      or optimized. If True, absorbed energy is an observable quantity that can be
                                      monitored or optimized. Default is False.
                                      
        - obs_ellipsiometric (bool): Flag to determine whether ellipsometric parameters should be observed or optimized.
                                     If True, ellipsometric parameters (Psi and Delta) are included as observables
                                     that can be monitored or optimized during the simulation or design process.
                                     Default is False.
                                     
        - obs_poynting (bool): Flag to determine whether the Poynting vector (representing the directional energy flow)
                               should be observed or optimized. If True, the Poynting vector is included as an observable
                               quantity that can be monitored or optimized. Default is False.
                               
        - *args: Additional positional arguments that can be passed to the class.
                 
        - **kwargs: Additional keyword arguments that can be passed to the class.
        """       
        self._thicknesses = None  # Store layer thicknesses.
        self._material_distribution = None  # Store material distribution.
        self._fixed_material_distribution = fixed_material_distribution  # Fixed material distribution flag.
        self._incoherency_list = None # Store incoherency as a boolean list.
        self._auto_coherency = auto_coherency  # Auto-coherency flag.
        self._any_incoherent = any_incoherent # Incoherency flag.
        self._incoming_medium = interpolate_nk(incoming_medium) # Incoming medium nk function.
        self._outgoing_medium = interpolate_nk(outgoing_medium)  # Outgoing medium nk function.
        self._is_material_set = False # Material setting flag.
        self._num_of_materials = None # Number of selected materials.
        self._obs_absorbed_energy = obs_absorbed_energy # Flag for absorbed energy is observable (optimizable) or not.
        self._obs_ellipsiometric = obs_ellipsiometric # Flag for ellipsometric parameters are observable (optimizable) or not.
        self._obs_poynting = obs_poynting# Flag for the Poynting vector is observable (optimizable) or not.
        
        self._nk_funcs = None
        
        # Handling fixed material distribution
        if not self._fixed_material_distribution:
            """
            In kwargs:
            - material_set (List[str], optional): List of material names, required if distribution is not fixed.
            """
            if "material_set" not in kwargs:
                raise ValueError("The 'material_set' key is missing from the input arguments.")
            material_set = kwargs.get("material_set", [])
            self._material_set = material_set  # Store material set
            self._is_material_set = True # Material set is not None
            self._num_of_materials = len(self._material_set)
            # Enumerate materials
            self._material_enum = self._enumerate_materials(self._material_set)  # Material to integer mapping
            # Initialize nk functions dictionary using iterate_nk_data
            self._nk_funcs = self._create_nk_funcs(interpolate_nk)  # Initialize nk functions

        # Set initial theta array as None
        self._layer_angles = None
        
        # Set kz array as None
        self._kz = None
        
        # Set _rt array as None
        self._rt = None
        
        # Set polarization array as None        
        self._polarization = None

    def _scan_material_set(self, material_distribution: List[str]) -> List[str]:
        """
        Scan the material distribution to determine the unique set of materials.
        
        Args:
        - material_distribution (List[str]): The material distribution list.
        
        Returns:
        - List[str]: A list of unique materials found in the distribution.
        """
        # Create a set to identify unique materials and then convert it to a list
        unique_materials = list(set(material_distribution))
        return unique_materials

    def _enumerate_materials(self, material_set: List[str]) -> Dict[int, str]:
        """
        Enumerate materials to create a mapping from material name to integer index.
        
        Args:
        - material_set (List[str]): The set of materials to enumerate.
        
        Returns:
        - Dict[str, int]: A dictionary mapping integer indices to material names.
        """
        # Map material names to indices
        return {material: i for i, material in enumerate(material_set)}

    def _enumerate_material_keys(self, material_set: List[str]) -> Dict[int, str]:
        """
        Enumerate material keys to create a mapping from integer index to material name.
        
        Args:
        - material_set (List[str]): The set of materials to enumerate.
        
        Returns:
        - Dict[int, str]: A dictionary mapping integer indices to material names.
        """
        # Map material indices to their names
        return {i: material for i, material in enumerate(material_set)}

    def _create_nk_funcs(self, interpolate_nk: Callable) -> Dict[int, Callable]:
        """
        Create a dictionary of nk functions for each material.

        Args:
        - iterate_nk_data (Callable): Function that returns an nk function for a given material.
        
        Returns:
        - Dict[int, Callable]: A dictionary mapping material index to nk function.
        """
        # Map material indices to their respective nk function based on material names
        return {i: interpolate_nk(material) for i, material in enumerate(self._material_set)}


    def _compute_kz_one_wl(self, 
                           nk_list: jnp.ndarray, # Array of complex refractive indices for different wavelengths
                           angle_index: Union[int, jnp.ndarray], # Index of angle of incidence for each layer
                           wavelength_index: Union[int, jnp.ndarray], # Index of wavelength array
                           wavelength: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the z-component of (complex) angular wavevector (kz)
        (just for 1 wl and init theta nk value).
    
        Args:
            nk_list (Array): A one-dimensional JAX array representing the complex refractive indices 
                            of the materials in each layer of the multilayer thin film. Each element 
                            is of the form `n + j*k`, where `n` is the refractive index, and `k` 
                            is the extinction coefficient, which accounts for the absorption in the material.
                            
                                                 
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
        # Calculate the z-component of the wave vector for each wavelength and angle
        return 2 * jnp.pi * nk_list[wavelength_index, :] * jnp.cos(self._layer_angles[angle_index, wavelength_index, :]) / jnp.array(wavelength)[wavelength_index]



    def _compute_kz(self,
                   layer_angles: Union[float, jnp.ndarray], 
                   wavelength: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Calculates the z-component of (complex) angular wavevector (kz) for a set of refractive indices (nk_list_2d) 
        and an initial angle of incidence (layer_angles) using vectorization.
    
        Args:
            layer_angles (Union[float, jnp.ndarray]): The initial angle of incidence (in radians). Can be 
                                                      a single float or a 1D/2D jax array (ndarray) depending 
                                                      on the use case.
            wavelength (Union[float, jnp.ndarray]): The wavelength or an array of wavelengths (ndarray) 
                                                   for which the computation will be performed.
    
        Returns:
            jnp.ndarray: A 3D JAX array where the [i, j, :] entry represents the angles of incidence 
                         for the j-th initial angle at the i-th wavelength. The size of the third dimension 
                         corresponds to the number of layers.
        """
        
        # Create a function that retrieves the refractive indices for each material in the distribution
        def get_nk_values(wl):
            # For each material in the distribution, call the corresponding nk function with the given wavelength
            return jnp.array([self._nk_funcs[mat_idx](wl) for mat_idx in self._material_distribution])
    
        # Use vmap to vectorize the get_nk_values function over the wavelength dimension
        # This will return a 2D array where each row corresponds to the refractive indices at a given wavelength
        nk_list_2d = vmap(get_nk_values)(wavelength)
        _theta_indices = jnp.arange(0,jnp.size(layer_angles), dtype = int) # Array or single value for the indices of angle of incidence
        _wavelength_indices = jnp.arange(0,jnp.size(wavelength), dtype = int) # Array or single value for  the indices of wavelength
        # Vectorize the _compute_layer_angles_one_wl function over the wavelength dimension (first dimension of nk_list_2d)
        # in_axes=(0, None, 0) means:
        # - The first argument (nk_list_2d) will not be vectorized
        # - The second argument (layer_angles) will be vectorized over the first dimension
        vmap_compute_kz = vmap(vmap(self._compute_kz_one_wl, (None, None, 0, None)), (None, 0, None, None))

        # Apply the vectorized function to get the 3D array of angles
        # The resulting array has dimensions (number_of_wavelengths, number_of_init_angles, number_of_layers)
        return vmap_compute_kz(nk_list_2d, _theta_indices, _wavelength_indices, wavelength)

    def _fresnel_s(self, _first_layer_n: Union[float, jnp.ndarray], _second_layer_n: Union[float, jnp.ndarray],
                   _first_layer_theta: Union[float, jnp.ndarray], _second_layer_theta: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        This function calculates the Fresnel reflection (r_s) and transmission (t_s) coefficients 
        for s-polarized light (electric field perpendicular to the plane of incidence) at the interface 
        between two materials. The inputs are the refractive indices and the angles of incidence and 
        refraction for the two layers.
    
        Args:
            _first_layer_n (Union[float, jnp.ndarray]): Refractive index of the first layer (incident medium). 
                Can be a float or an array if computing for multiple incident angles/materials.
            _second_layer_n (Union[float, jnp.ndarray]): Refractive index of the second layer (transmitted medium). 
                Similar to the first argument, this can also be a float or an array.
            _first_layer_theta (Union[float, jnp.ndarray]): Angle of incidence in the first layer (in radians). 
                Can be a float or an array.
            _second_layer_theta (Union[float, jnp.ndarray]): Angle of refraction in the second layer (in radians). 
                Can be a float or an array.
    
        Returns:
        - Array[jnp.ndarray, jnp.ndarray]: A jax array containing:
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
        # Calculate the reflection coefficient for s-polarized light using Fresnel's equations.
        # The formula: r_s = (n1 * cos(theta1) - n2 * cos(theta2)) / (n1 * cos(theta1) + n2 * cos(theta2))
        # This measures how much of the light is reflected at the interface.
        r_s = ((_first_layer_n * jnp.cos(_first_layer_theta) - _second_layer_n * jnp.cos(_second_layer_theta)) /
               (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))
        
        # Calculate the transmission coefficient for s-polarized light using Fresnel's equations.
        # The formula: t_s = 2 * n1 * cos(theta1) / (n1 * cos(theta1) + n2 * cos(theta2))
        # This measures how much of the light is transmitted through the interface.
        t_s = (2 * _first_layer_n * jnp.cos(_first_layer_theta) /
               (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))
        
        # Return the reflection and transmission coefficients as a JAX array
        return jnp.array([r_s, t_s])
    
    def _fresnel_p(self, _first_layer_n: Union[float, jnp.ndarray], _second_layer_n: Union[float, jnp.ndarray],
                   _first_layer_theta: Union[float, jnp.ndarray], _second_layer_theta: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        This function calculates the Fresnel reflection (r_p) and transmission (t_p) coefficients 
        for p-polarized light at the interface between two different media. It uses the refractive indices
        of the two media (_first_layer_n and _second_layer_n) and the incident and transmitted angles
        (_first_layer_theta and _second_layer_theta) to compute these values.
    
        Args:
            _first_layer_n: Refractive index of the first medium (can be float or ndarray).
            _second_layer_n: Refractive index of the second medium (can be float or ndarray).
            _first_layer_theta: Incident angle (in radians) in the first medium (can be float or ndarray).
            _second_layer_theta: Transmitted angle (in radians) in the second medium (can be float or ndarray).
    
        Returns:
        - Array[jnp.ndarray, jnp.ndarray]: A jax array containing:
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
        # Calculate the reflection coefficient for p-polarized light (r_p)
        # This equation is based on the Fresnel equations for p-polarization, where 
        # r_p is the ratio of the reflected and incident electric field amplitudes for p-polarized light.
        r_p = ((_second_layer_n * jnp.cos(_first_layer_theta) - _first_layer_n * jnp.cos(_second_layer_theta)) /
               (_second_layer_n * jnp.cos(_first_layer_theta) + _first_layer_n * jnp.cos(_second_layer_theta)))
    
        # Calculate the transmission coefficient for p-polarized light (t_p)
        # This equation is also derived from the Fresnel equations for p-polarization.
        # t_p represents the ratio of the transmitted and incident electric field amplitudes.
        t_p = (2 * _first_layer_n * jnp.cos(_first_layer_theta) /
               (_second_layer_n * jnp.cos(_first_layer_theta) + _first_layer_n * jnp.cos(_second_layer_theta)))
    
        # Return the reflection and transmission coefficients as a tuple of jnp arrays
        # Both r_p and t_p are essential for understanding how light interacts with different layers.
        return jnp.array([r_p, t_p])
    
    # def _interface(self,
    #     _any_incoherent: bool,
    #     _polarization: bool,
    #     _first_layer_theta: Union[float, jnp.ndarray],
    #     _second_layer_theta: Union[float, jnp.ndarray],
    #     _first_layer_n: Union[float, jnp.ndarray],
    #     _second_layer_n: Union[float, jnp.ndarray]
    # ) -> Union[jnp.ndarray, Tuple[float, float]]:
    #     """
    #     Calculate reflectance and transmittance at the interface between two layers
    #     using Fresnel equations, considering the polarization of light and whether
    #     the layers are coherent or incoherent.
    
    #     Parameters
    #     ----------
    #     _any_incoherent : bool
    #         Flag indicating if any of the layers at the boundary are incoherent.
    #         - False: Both boundaries are coherent.
    #         - True: At least one boundary is incoherent.
        
    #     _polarization : Optional[bool]
    #         Polarization state of the incoming light.
    #         - None: Both s and p polarizations (unpolarized light).
    #         - False: s-polarization.
    #         - True: p-polarization.
        
    #     _first_layer_theta : Union[float, jnp.ndarray]
    #         Angle of incidence with respect to the normal at the boundary of the first layer.
        
    #     _second_layer_theta : Union[float, jnp.ndarray]
    #         Angle of refraction with respect to the normal at the boundary of the second layer.
        
    #     _first_layer_n : Union[float, jnp.ndarray]
    #         Refractive index of the first layer.
        
    #     _second_layer_n : Union[float, jnp.ndarray]
    #         Refractive index of the second layer.
        
    #     Returns
    #     -------
    #     Union[jnp.ndarray, Tuple[float, float]]
    #         - If polarization is None (both s and p), return a 2x2 matrix:
    #           - Coherent: [[r_s, r_p], [t_s, t_p]]
    #           - Incoherent: [[R_s, R_p], [T_s, T_p]]
    #         - If polarization is False, return a vector [r_s, t_s] or [R_s, T_s].
    #         - If polarization is True, return a vector [r_p, t_p] or [R_p, T_p].
    #     """
    #     # Handle the incoherent case
    #     if _any_incoherent:
    #         if _polarization is None:
    #             # Unpolarized light: both s and p polarizations
    #             _r_s, _t_s = self._fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
    #             _r_p, _t_p = self._fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
    #             _R_s = jnp.abs(_r_s) ** 2
    #             _T_s = jnp.abs(_t_s) ** 2
    #             _R_p = jnp.abs(_r_p) ** 2
    #             _T_p = jnp.abs(_t_p) ** 2
    #             return jnp.array([[_R_s, _R_p], [_T_s, _T_p]])
    #         elif _polarization is False:
    #             _r_s, _t_s = self._fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
    #             _R_s = jnp.abs(_r_s) ** 2
    #             _T_s = jnp.abs(_t_s) ** 2
    #             return jnp.array([_R_s, _T_s])
    #         elif _polarization is True:
    #             _r_p, _t_p = self._fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
    #             _R_p = jnp.abs(_r_p) ** 2
    #             _T_p = jnp.abs(_t_p) ** 2
    #             return jnp.array([_R_p, _T_p])
    
    #     # Handle the coherent case
    #     else:
    #         if _polarization is None:
    #             # Unpolarized light: both s and p polarizations
    #             _r_s, _t_s = self._fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
    #             _r_p, _t_p = self._fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
    #             return jnp.array([[_r_s, _t_s], [_r_p, _t_p]])
    #         elif _polarization is False:
    #             _r_s, _t_s = self._fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
    #             return jnp.array([_r_s, _t_s])
    #         elif _polarization is True:
    #             _r_p, _t_p = self._fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
    #             return jnp.array([_r_p, _t_p])



    def _compute_rt_one_wl(self,
                           angle_index: Union[int, jnp.ndarray],
                           wavelength_index: Union[int, jnp.ndarray],
                           wavelength: Union[float, jnp.ndarray], 
                           polarization: bool) -> jnp.ndarray:
        """
        Computes the reflectance and transmittance for a single wavelength 
        across multiple layers in a stack of materials. The computation 
        takes into account the refractive index of each layer, the angle of 
        incidence in each layer, the wavelength of the light, and the 
        polarization of the light.
    
        Args:
            wavelength (float or jnp.ndarray): The wavelength of light, given as either 
                                               a scalar or a JAX array.
            polarization (bool): Boolean flag that determines the polarization state of the light. 
                                 If False, s-polarization is used; if True, p-polarization is used.
    
        Returns:
            jnp.ndarray: A 1D JAX array representing the reflectance and transmittance 
                         coefficients at the specified wavelength and polarization.
        """
        
        # Create a function that retrieves the refractive indices for each material in the distribution
        def get_nk_values(wl):
            # For each material in the distribution, call the corresponding nk function with the given wavelength
            return jnp.array([self._nk_funcs[mat_idx](wl) for mat_idx in self._material_distribution])
    

        nk_list = vmap(get_nk_values)(wavelength)
        layer_angles = self._layer_angles[angle_index, wavelength_index, :]

        # Initialize the state for `jax.lax.scan`. The first element (0) is a placeholder 
        # and won't be used. The second element is a 2D array of zeros to hold intermediate 
        # results, representing the reflectance and transmittance across layers.
        init_state = (0, jnp.zeros((len(nk_list) - 2, 2), dtype=jnp.float32))  # Initial state with an array of zeros
        # The shape of `jnp.zeros` is (num_layers - 2, 2) because we exclude the first 
        # and last layers, assuming they are boundary layers.
    
        # Stack the refractive indices (`nk_list`) for each adjacent pair of layers.
        # This creates a new array where each element contains a pair of adjacent refractive indices 
        # from `nk_list`, which will be used to compute the reflection and transmission at the interface 
        # between these two layers.
        stacked_nk_list = jnp.stack([nk_list[:-2], nk_list[1:-1]], axis=1)  # Stack the original and shifted inputs for processing in pairs
        # For example, if `nk_list` is [n1, n2, n3, n4], this will create pairs [(n1, n2), (n2, n3), (n3, n4)].
    
        # Similarly, stack the angles for adjacent layers.
        # The same logic applies to `layer_angles` as for `nk_list`. Each pair of adjacent layers 
        # will have an associated pair of angles.
        stacked_layer_angles = jnp.stack([layer_angles[:-2], layer_angles[1:-1]], axis=1)
        # This operation aligns the angles with the corresponding refractive indices.
    
        # Now we need to compute reflectance and transmittance for each interface. 
        # This can be done using `jax.lax.scan`, which efficiently loops over the stacked pairs 
        # of refractive indices and angles.
    
        # If the light is s-polarized (polarization = False), we call the function `_compute_rt_at_interface_s`.
        # This function calculates the reflection and transmission coefficients specifically for s-polarized light.
        if polarization == False:
            rt_one_wl, _ = jax.lax.scan(_compute_rt_at_interface_s, init_state, (stacked_nk_list, stacked_layer_angles))  # s-polarization case
            # `jax.lax.scan` applies the function `_compute_rt_at_interface_s` to each pair of adjacent layers 
            # along with the corresponding angles. It processes this in a loop, accumulating the results.
    
        # If the light is p-polarized (polarization = True), we use `_compute_rt_at_interface_p` instead.
        # This function handles p-polarized light.
        elif polarization == True:
            rt_one_wl, _ = jax.lax.scan(_compute_rt_at_interface_p, init_state, (stacked_nk_list, stacked_layer_angles))  # p-polarization case
            # The same process as above but with a function specific to p-polarized light.
    
        # Finally, return the computed reflectance and transmittance coefficients. 
        # The result is stored in `rt_one_wl[1]` (the second element of `rt_one_wl`), which corresponds 
        # to the reflectance and transmittance after all layers have been processed.
        return rt_one_wl[1]  # Return a 1D theta array for each layer
        # This output is the desired result: the reflectance and transmittance for the given wavelength.





    def _compute_rt(self,
                   wavelength: Union[float, jnp.ndarray],
                   polarization: bool) -> jnp.ndarray:
 

        angle_indices = jnp.arange(0,jnp.size(self._material_distribution), dtype = int) # Array or single value for the indices of angle of incidence
        wavelength_indices = jnp.arange(0,jnp.size(wavelength), dtype = int) # Array or single value for  the indices of wavelength
        # Vectorize the _compute_layer_angles_one_wl function over the wavelength dimension (first dimension of nk_list_2d)
        # in_axes=(0, None, 0) means:
        # - The first argument (nk_list_2d) will not be vectorized
        # - The second argument (initial_theta) will be vectorized over the first dimension
        vmap_compute_rt = vmap(vmap(self._compute_rt_one_wl, (0, None, None, None)), (None, 0, None, None))

        # Apply the vectorized function to get the 3D array of angles
        # The resulting array has dimensions (number_of_wavelengths, number_of_init_angles, number_of_layers)
        return vmap_compute_rt(angle_indices, wavelength_indices, wavelength, polarization)

    # Getter for thicknesses
    @property
    def thicknesses(self) -> List[float]:
        """
        Get the list of layer thicknesses.

        Returns:
        - List[float]: List of thicknesses.
        """
        return self._thicknesses

    # Setter for dlist
    @thicknesses.setter
    def thicknesses(self, new_thicknesses: jnp.ndarray) -> None:
        """
        Set the list of layer thicknesses.

        Args:
        - new_thicknesses (jnp.ndarray): New list of thicknesses.
        """
        self._thicknesses = new_thicknesses
        # Initialize incoherency list based on auto_coherency flag
        if self._auto_coherency:
            # Determine incoherency based on dlist values if auto_coherency is True and no list is provided
            self._incoherency_list = determine_coherency(self._thicknesses)
        else:
            if self._incoherency_list is None:
                # If current '_incoherency_list' is 'None', it is assumed that every layer is coherent
                self._incoherency_list = [False] * len(self._thicknesses)
        self._any_incoherent = any(self._incoherency_list)
            
    # Getter for material_distribution
    @property
    def material_distribution(self) -> List[str]:
        """
        Get the material distribution list.

        Returns:
        - List[str]: List of material names in the distribution.
        """
        return self._material_distribution

    # Setter for material_distribution
    def set_material_distribution(self, material_info: tuple) -> None:
        """
        Set the material distribution list and update theta.

        Args:
            material_info (tuple): Tuple containing the new list of new_material_distribution.
                - new_material_distribution (List[int]): Distribution of materials in the stack.
                - theta (Union[float, jnp.ndarray]): Incoming light theta array to be used for inner angle calculation.
                - wavelength (Union[float, jnp.ndarray]): Incoming light wavelength array.
        """
        new_material_distribution, theta, wavelength, polarization = material_info
        # Check if the lengths of the provided lists are consistent
        if len(self._thicknesses) != len(new_material_distribution):
            raise ValueError("Length of initial_thicknesses and new_material_distribution must be the same.")
        if self._fixed_material_distribution and self._is_material_set:
            raise ValueError("Material distribution is fixed and cannot be reassigned.")
        self._material_distribution = new_material_distribution
        
        if not self._is_material_set:
            if any(isinstance(item, int) for item in new_material_distribution):
                raise ValueError("The list contains numbers, which is not allowed in the first assignment of the materials.")
            self._material_set = self._scan_material_set(new_material_distribution)
            self._is_material_set = True # Material set is not None
            self._num_of_materials = len(self._material_set)
            # Enumerate materials
            self._material_enum = self._enumerate_materials(self._material_set)  # Material to integer mapping
            self._material_enum_keys = self._enumerate_material_keys(self._material_set)  # Integer to material mapping
            self._material_distribution = [int(self._material_enum[material]) for material in new_material_distribution]
            # Initialize nk functions dictionary using iterate_nk_data
            self._nk_funcs = self._create_nk_funcs(interpolate_nk)  # Initialize nk functions
            self._polarization = polarization
        self._layer_angles = compute_layer_angles(nk_functions = self._nk_funcs, material_distribution = self._material_distribution,
                                           initial_theta = theta, wavelength = wavelength, polarization = self._polarization,
                                           incoming_medium = self._incoming_medium, outgoing_medium = self._outgoing_medium)
        print((self._layer_angles).shape)       
        self._kz = self._compute_kz(nk_functions = self._nk_funcs, material_distribution = self._material_distribution,
                                   initial_theta = theta, wavelength = wavelength)    
        print((self._kz).shape)
        if self._polarization is None:
            self._rt = (self._compute_rt(wavelength = wavelength, polarization=False),
                        self._compute_rt(wavelength = wavelength, polarization=True))
        else:
            self._rt = self._compute_rt(wavelength = wavelength, polarization=self._polarization)
            

    # Getter for incoherency_list
    @property
    def incoherency_list(self) -> List[bool]:
        """
        Get the list of incoherency flags.

        Returns:
        - List[bool]: List of incoherency flags for each layer.
        """
        return self._incoherency_list

    # Setter for incoherency_list
    @incoherency_list.setter
    def incoherency_list(self, new_incoherency_list: List[bool]) -> None:
        """
        Set the list of incoherency flags.

        Args:
        - new_incoherency_list (List[bool]): New list of incoherency flags.
        """
        # Ensure the length of the new list matches the length of dlist
        if len(new_incoherency_list) != len(self._thicknesses):
            raise ValueError("Incoherency list must have the same length as dlist.")
        if not self._auto_coherency:
            self._incoherency_list = new_incoherency_list
        else:
            raise ValueError("Auto coherency is True, you cannot change incoherency list.")
    
    # Getter for theta
    @property
    def theta(self) -> jnp.ndarray:
        """
        Get the list of theta jnp array.

        Returns:
        - List[bool]: jax.numpy array of theta for each layer.
        """
        return self._layer_angles

    # Getter for kz
    @property
    def kz(self) -> jnp.ndarray:
        """
        Get the list of theta jnp array.

        Returns:
        - List[bool]: jax.numpy array of kz for each layer, wl and theta.
        """
        return self._kz

    # Setter for incoherency_list
    @incoherency_list.setter
    def incoherency_list(self, new_incoherency_list: List[bool]) -> None:
        """
        Set the list of incoherency flags.

        Args:
        - new_incoherency_list (List[bool]): New list of incoherency flags.
        """
        # Ensure the length of the new list matches the length of dlist
        if len(new_incoherency_list) != len(self._thicknesses):
            raise ValueError("Incoherency list must have the same length as dlist.")
        if not self._auto_coherency:
            self._incoherency_list = new_incoherency_list
        else:
            raise ValueError("Auto coherency is True, you cannot change incoherency list.")

    # Getter for fixed_material_distribution
    @property
    def fixed_material_distribution(self) -> bool:
        """
        Get the any_incoherent boolean.

        Returns:
        - [bool]:Boolean for the purpose whether there are any incoherent layers in the stack.
        """
        return self._fixed_material_distribution

    # Getter for any_incoherent
    @property
    def any_incoherent(self) -> bool:
        """
        Get the any_incoherent boolean.

        Returns:
        - [bool]:Boolean for the purpose whether there are any incoherent layers in the stack.
        """
        return self._any_incoherent
    
    # Getter for incoming_medium nk function
    @property
    def incoming_medium(self) -> Callable:
        """
        Get the incoming medium nk data.

        Returns:
        - (Callable) : Integer of incoming_medium index for incoming nk data function
        """
        return self._incoming_medium

    # Getter for outgoing_medium nk function
    @property
    def outgoing_medium(self) -> Callable:
        """
        Get the outgoing medium nk data.

        Returns:
        - (Callable) : Integer of _outgoing_medium index for outgoing nk data
        """
        return self._outgoing_medium

    # Getter for rt array
    @property
    def rt(self) -> jnp.ndarray:
        """
        Get the rt array.

        Returns:
        - (jnp.ndarray) : Array of _outgoing_medium index for rt array
        """
        return self._rt

    # Getter for obs_absorbed_energy
    @property
    def obs_absorbed_energy(self) -> bool:
        """
        Get the obs_absorbed_energy boolean.

        Returns:
        - [bool]:Boolean for absorbed energy is observable (optimizable) or not.
        """
        return self._obs_absorbed_energy
    
    # Getter for obs_ellipsiometric
    @property
    def obs_ellipsiometric(self) -> bool:
        """
        Get the any_incoherent boolean.

        Returns:
        - [bool]:Boolean for ellipsometric parameters are observable (optimizable) or not.
        """
        return self._obs_ellipsiometric
    
    # Getter for obs_poynting
    @property
    def obs_poynting(self) -> bool:
        """
        Get the any_incoherent boolean.

        Returns:
        - [bool]:Boolean for the Poynting vector is observable (optimizable) or not.
        """
        return self._obs_poynting

    # Getter for num_of_materials
    @property
    def num_of_materials(self) -> int:
        """
        Get the number of of the materials in material_set

        Returns:
        - (int) : Int for number of materials in the materal_set
        """
        return self._num_of_materials
    
    # Getter for nk_funcs
    @property
    def nk_funcs(self) -> Dict[int, Callable]:
        """
        Get the dictionary for the nk material data functions

        Returns:
        - (Dict[int, Callable]) : Dict for nk material data functions
        """
        return self._nk_funcs
    
    def save_log(self, filename: str):
        """
        Save the multilayer stack structure to a csv file, including details about each layer and its properties.
    
        Args:
            filename (str): The name of the file to which the csv log will be saved. This file will contain details about
                            the multilayer thin film stack, including material names, thicknesses, and coherency states.
    
        Returns:
            None: This function does not return any value. It creates a csv file with the specified filename, storing
                  the multilayer stack details in a tabular format.
        
        The CSV file will have the following structure:
        - Header: ['layer material', 'thickness', 'coherency']
        - Rows:
            1. The first row represents the incoming medium with a thickness from the first element of `self._thicknesses`
               and a coherency state of "i" (incoherent).
            2. Intermediate rows represent each layer in the stack, including the material name, thickness, and coherency
               state derived from `self._incoherency_list`. Coherency is denoted as "i" (incoherent) for `True` and "c"
               (coherent) for `False`.
            3. The last row represents the outgoing medium with infinite thickness and a coherency state of "i" (incoherent).
        """
        # Prepare the data to be written to the CSV file
        data = []
    
        # First row: Incoming medium (always incoherent, thickness is that of the first material)
        first_row = [self.index_to_material[0], self._thicknesses[0], "i"]
        data.append(first_row)
    
        # Intermediate rows: Add material data from index_to_material
        for i in range(1, len(self._thicknesses) - 1):
            material = self._material_enum_keys[i]
            thickness = self._thicknesses[i]
            coherency = "i" if self._incoherency_list[i] else "c"
            row = [material, thickness, coherency]
            data.append(row)
    
        # Last row: Outgoing medium (always incoherent, thickness is infinity)
        last_row = [self.index_to_material[-1], float('inf'), "i"]
        data.append(last_row)
    
        # Convert the data to a NumPy array
        data_array = np.array(data, dtype=object)
    
        # Save the data to a CSV file using NumPy
        header = 'layer material,thickness,coherency'
        np.savetxt(filename, data_array, delimiter=',', fmt='%s', header=header, comments='')


def determine_coherency(thicknesses: jnp.ndarray) -> List[bool]:
    """
    Determine the incoherency of layers based on their thickness.

    Args:
    - thicknesses (jnp.ndarray): Array of thicknesses for the layers.

    Returns:
    - List[bool]: List indicating incoherency (True if incoherent, False if coherent).
    """
    threshold = 300.0  # Threshold in microns
    d_squared = jnp.square(thicknesses)  # Compute the square of thicknesses
    # Incoherency if the squared thickness exceeds the threshold
    incoherency_list = jnp.greater(d_squared, threshold)
    return list(incoherency_list)  # Convert the result back to a list of booleans

