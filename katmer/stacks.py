import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
from typing import Callable, List, Dict, Union, Tuple


from katmer.data import interpolate_nk
from katmer.light import compute_layer_angles

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
    
    def _fresnel_p(self, _first_layer_n: Union[float, jnp.ndarray], _second_layer_n: Union[float, jnp.ndarray],
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
    
    def _interface(self,
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
                _r_s, _t_s = self._fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
                _r_p, _t_p = self._fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
                _R_s = jnp.abs(_r_s) ** 2
                _T_s = jnp.abs(_t_s) ** 2
                _R_p = jnp.abs(_r_p) ** 2
                _T_p = jnp.abs(_t_p) ** 2
                return jnp.array([[_R_s, _R_p], [_T_s, _T_p]])
            elif _polarization is False:
                _r_s, _t_s = self._fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
                _R_s = jnp.abs(_r_s) ** 2
                _T_s = jnp.abs(_t_s) ** 2
                return jnp.array([_R_s, _T_s])
            elif _polarization is True:
                _r_p, _t_p = self._fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
                _R_p = jnp.abs(_r_p) ** 2
                _T_p = jnp.abs(_t_p) ** 2
                return jnp.array([_R_p, _T_p])
    
        # Handle the coherent case
        else:
            if _polarization is None:
                # Unpolarized light: both s and p polarizations
                _r_s, _t_s = self._fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
                _r_p, _t_p = self._fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
                return jnp.array([[_r_s, _t_s], [_r_p, _t_p]])
            elif _polarization is False:
                _r_s, _t_s = self._fresnel_s(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
                return jnp.array([_r_s, _t_s])
            elif _polarization is True:
                _r_p, _t_p = self._fresnel_p(_first_layer_n, _second_layer_n, _first_layer_theta, _second_layer_theta)
                return jnp.array([_r_p, _t_p])

    def _compute_rt_at_interface(self, carry, concatenated_nk_list_theta):
        concatenated_nk_list, theta = concatenated_nk_list_theta
        carry_idx, carry_values = carry
        
        # Compute r,t
        r_t_matrix = self._interface(_any_incoherent = self._any_incoherent,_polarization = self._polarization,
                                _first_layer_theta = theta[0] ,
                                _second_layer_theta = theta[1],
                                _first_layer_n = concatenated_nk_list[0],
                                _second_layer_n = concatenated_nk_list[1])
        
        carry_values = carry_values.at[carry_idx, :].set(r_t_matrix)
        
        carry_idx = carry_idx + 1  # Move to the next index
        return (carry_idx, carry_values), None

    def _compute_rt_one_wl(self, nk_list: jnp.ndarray, theta_index: Union[int, jnp.ndarray], 
                           wavelength_index: Union[int, jnp.ndarray], wavelength: Union[float, jnp.ndarray]) -> jnp.ndarray:
        incoming_medium = jnp.expand_dims(self._incoming_medium(wavelength[wavelength_index]), axis=0)
        nk_values = nk_list[wavelength_index, :]
        outgoing_medium = jnp.expand_dims(self.outgoing_medium(wavelength[wavelength_index]), axis=0)

        concatenated_nk_list = jnp.concatenate([incoming_medium, nk_values, outgoing_medium], axis=0)

        #concatenated_nk_list = jnp.concatenate([self._incoming_medium(wavelength[wavelength_index]), nk_list[wavelength_index, :], self.outgoing_medium(wavelength[wavelength_index])])
        if self._polarization == None:
            init_state = (0, jnp.zeros((len(nk_values)+1, 2, 2), dtype=jnp.float32))  # Initial state with an array of zeros
        else:
            init_state = (0, jnp.zeros((len(nk_values)+1, 2), dtype=jnp.float32))  # Initial state with an array of zeros
        
        # Create shifted versions of inputs1 and inputs2 with an extra zero at the end
        padded_concatenated_nk_list = jnp.pad(concatenated_nk_list, (0, 1), constant_values=0)
        print(padded_concatenated_nk_list.shape)
        padded_theta = jnp.pad(self._layer_angles[theta_index, wavelength_index, :], (0, 1), constant_values=0)
        print(padded_concatenated_nk_list.shape)
        # Stack the original and shifted inputs for processing in pairs
        stacked_nk_list = jnp.stack([concatenated_nk_list, padded_concatenated_nk_list[1:]], axis=1)
        stacked_theta = jnp.stack([self._layer_angles[theta_index, wavelength_index, :], padded_theta[1:]], axis=1)        
        print(stacked_nk_list.shape)
        print(stacked_theta.shape)
        # Use jax.lax.scan to iterate over inputs1 and inputs2
        #rt_one_wl, _ = jax.lax.scan(lambda carry, ntheta: self._compute_rt_at_interface(carry, ntheta[0], ntheta[1]),
        #                          init_state, (stacked_nk_list, stacked_theta))        

        # Now use this function in `jax.lax.scan`
        rt_one_wl, _ = jax.lax.scan(self._compute_rt_at_interface, init_state, (stacked_nk_list, stacked_theta))


        # Return a 1D theta array for each layer
        return rt_one_wl[1]
    
    def _compute_rt(self, nk_functions: Dict[int, Callable],
                   material_distribution: List[int], 
                   initial_theta: Union[float, jnp.ndarray], 
                   wavelength: Union[float, jnp.ndarray]) -> jnp.ndarray:
 
        
        # Create a function that retrieves the refractive indices for each material in the distribution
        def get_nk_values(wl):
            # For each material in the distribution, call the corresponding nk function with the given wavelength
            return jnp.array([nk_functions[mat_idx](wl) for mat_idx in material_distribution])
    
        # Use vmap to vectorize the get_nk_values function over the wavelength dimension
        # This will return a 2D array where each row corresponds to the refractive indices at a given wavelength
        nk_list_2d = vmap(get_nk_values)(wavelength)
        _theta_indices = jnp.arange(0,jnp.size(initial_theta), dtype = int) # Array or single value for the indices of angle of incidence
        _wavelength_indices = jnp.arange(0,jnp.size(wavelength), dtype = int) # Array or single value for  the indices of wavelength
        # Vectorize the _compute_layer_angles_one_wl function over the wavelength dimension (first dimension of nk_list_2d)
        # in_axes=(0, None, 0) means:
        # - The first argument (nk_list_2d) will not be vectorized
        # - The second argument (initial_theta) will be vectorized over the first dimension
        vmap_compute_rt = vmap(vmap(self._compute_rt_one_wl, (None, None, 0, None)), (None, 0, None, None))

        # Apply the vectorized function to get the 3D array of angles
        # The resulting array has dimensions (number_of_wavelengths, number_of_init_angles, number_of_layers)
        return vmap_compute_rt(nk_list_2d, _theta_indices, _wavelength_indices, wavelength)

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
        self._rt = self._compute_rt(nk_functions = self._nk_funcs, material_distribution = self._material_distribution,
                                   initial_theta = theta, wavelength = wavelength) 
        print((self._rt).shape)  

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