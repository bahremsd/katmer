import jax.numpy as jnp
from jax import vmap
import numpy as np
from typing import Callable, List, Dict, Union


from autodiffthinfilms.data import interpolate_nk
from autodiffthinfilms.light import compute_layer_angles

class Stack:
    """
    Stack class to simulate multilayer thin films using the Transfer Matrix Method (TMM).
    This class is designed for inverse design simulations and includes advanced features for
    handling material properties, layer thicknesses, and incoherency options.
    """

    def __init__(self, auto_coherency: bool = True, any_incoherent: bool = False,
                 fixed_material_distribution: bool = False, incoming_nk: Union[float, jnp.ndarray] = jnp.array(1 + 0j),
                 outgoing_nk: Union[float, jnp.ndarray] = jnp.array(1 + 0j), 
                 obs_absorbed_energy: bool = False, obs_ellipsiometric: bool = False,
                 obs_poynting: bool = False, *args, **kwargs):
        """
        Initialize the Stack class with material data, layer thicknesses, and coherency options.
        
        Args:
        - auto_coherency (bool): Flag to determine whether coherency should be automatically handled.
        - any_incoherent (bool): Flag to indicate whether there are any incoherent layers in the stack.
        - fixed_material_distribution (bool): Determines if material distribution is fixed.
        - incoming_nk (Union[float, jnp.ndarray]): This represents the nk information for the medium 
                      through which light enters the stack. Although this medium can be considered as a layer
                      with infinite thickness. Default is air.
        - outgoing_nk (Union[float, jnp.ndarray]): This represents the nk information for the medium 
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
        self._incoming_nk = incoming_nk # Incoming medium nk data.
        self._outgoing_nk = outgoing_nk  # Outgoing medium nk data.
        self._is_material_set = False # Material setting flag.
        self._num_of_materials = None # Number of selected materials.
        self._obs_absorbed_energy = obs_absorbed_energy # Flag for absorbed energy is observable (optimizable) or not.
        self._obs_ellipsiometric = obs_ellipsiometric # Flag for ellipsometric parameters are observable (optimizable) or not.
        self._obs_poynting = obs_poynting# Flag for the Poynting vector is observable (optimizable) or not.
        
        # Handling fixed material distribution
        if not self._fixed_material_distribution:
            """
            In kwargs:
            - material_set (List[str], optional): List of material names, required if distribution is not fixed.
            """
            material_set = kwargs.get("material_set", [])
            self._material_set = material_set  # Store material set
            self._is_material_set = True # Material set is not None
            self._num_of_materials = len(self._material_set)
            # Enumerate materials
            self._material_enum = self._enumerate_materials(self._material_set)  # Material to integer mapping
            # Initialize nk functions dictionary using iterate_nk_data
            self._nk_funcs = self._create_nk_funcs(interpolate_nk)  # Initialize nk functions

        # Set initial theta array as None
        self._theta = None
        
        # Set kz array as None
        self._kz = None

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


    def _compute_kz_one_wl(self, nk_list: jnp.ndarray, theta_index: Union[int, jnp.ndarray], 
                           wavelength_index: Union[int, jnp.ndarray], wavelength: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Computes the z-component of (complex) angular wavevector (kz)
        (just for 1 wl and init theta nk value).
    
        Args:
            nk_list (Array): A one-dimensional JAX array representing the complex refractive indices 
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
        # Return a 1D theta array for each layer
        return 2 * jnp.pi * nk_list[wavelength_index, :] * jnp.cos(self._theta[theta_index, wavelength_index, :]) / jnp.array(wavelength)[wavelength_index]
    
    def compute_kz(self, nk_functions: Dict[int, Callable],
                   material_distribution: List[int], 
                   initial_theta: Union[float, jnp.ndarray], 
                   wavelength: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Calculates the z-component of (complex) angular wavevector (kz) for a set of refractive indices (nk_list_2d) 
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
    
        Returns:
            jnp.ndarray: A 3D JAX array where the [i, j, :] entry represents the angles of incidence 
                         for the j-th initial angle at the i-th wavelength. The size of the third dimension 
                         corresponds to the number of layers.
        """
        
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
        vmap_compute_kz = vmap(self._compute_kz_one_wl, in_axes=(None, None, 0 ,0, None))
    
        # Apply the vectorized function to get the 3D array of angles
        # The resulting array has dimensions (number_of_wavelengths, number_of_init_angles, number_of_layers)
        return vmap_compute_kz(nk_list_2d, _theta_indices, _wavelength_indices, wavelength)


    def __iter__(self):
        """
        Iterator that yields the thickness, complex refractive index (n + jk), and theta for each layer.
    
        Each iteration provides:
        - d: The thickness of the current layer.
        - nk_func: A function that returns the complex refractive index for the current layer at a given wavelength.
        - theta_i: The angle of incidence/refraction at the current layer interface.
        - theta_ip1: The angle at the next layer interface. Defaults to 0 if the current layer is the last one.
        """
        # Iterate over the range of layers in the material stack
        for i in range(len(self._thicknesses)):
            # Retrieve the thickness of the current layer
            d = self._thicknesses[i]
            # Retrieve the function that calculates the complex refractive index for the current layer
            nk_func = self._nk_funcs[i]
            # Retrieve the angle at the current layer interface
            theta_i = self._theta[i]
            # Retrieve the angle at the next layer interface if it exists, otherwise default to 0
            theta_ip1 = self._theta[i+1] if i+1 < len(self._theta) else 0
            # Yield a tuple containing the thickness, refractive index function, and angles for this layer
            yield d, nk_func, theta_i, theta_ip1

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
    @material_distribution.setter
    def material_distribution(self, new_material_distribution: Union[List[int], List[str]], theta: Union[float, jnp.ndarray],
                              wavelength: Union[float, jnp.ndarray]) -> None:
        """
        Set the material distribution list and update theta.

        Args:
        - new_material_distribution (List[int]): Distribution of materials in the stack.
        - theta (Union[float, jnp.ndarray]): Incoming light theta array to be used for inner angle calculation.
        """
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
        self._theta = compute_layer_angles(nk_functions = self._nk_funcs, material_distribution = self._material_distribution,
                                           initial_theta = theta, wavelength = wavelength)       
        self._kz = self.compute_kz(nk_functions = self._nk_funcs, material_distribution = self._material_distribution,
                              initial_theta = theta, wavelength = wavelength)    

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
        return self._theta

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
    
    # Getter for incoming_nk
    @property
    def incoming_nk(self) -> Union[float, jnp.ndarray]:
        """
        Get the incoming medium nk data.

        Returns:
        - (Union[float, jnp.ndarray]) : Float or jax.numpy array of incoming_nk for incoming nk data
        """
        return self._incoming_nk

    # Getter for outgoing_nk
    @property
    def outgoing_nk(self) -> Union[float, jnp.ndarray]:
        """
        Get the outgoing medium nk data.

        Returns:
        - (Union[float, jnp.ndarray]) : Float or jax.numpy array of outgoing_nk for outgoing nk data
        """
        return self._incoming_nk

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
