import jax.numpy as jnp
from jax import jit
from typing import Callable, List, Dict

from autodiffthinfilms.data import interpolate_nk

class Stack:
    """
    Stack class to simulate multilayer thin films using the Transfer Matrix Method (TMM).
    This class is designed for inverse design simulations and includes advanced features for
    handling material properties, layer thicknesses, and incoherency options.
    """

    def __init__(self, initial_thicknesses: List[float],auto_coherency: bool = True,
                 initial_incoherency_list: List[bool] = None, 
                 fixed_material_distribution: bool = False, *args, **kwargs):
        """
        Initialize the Stack class with material data, layer thicknesses, and coherency options.
        
        Args:
        - initial_thicknesses (List[float]): Initial list of layer thicknesses.
        - auto_coherency (bool): Flag to determine whether coherency should be automatically handled.
        - initial_incoherency_list (List[bool], optional): Initial incoherency list, default is None.
        - fixed_material_distribution (bool): Determines if material distribution is fixed.
        
        Raises:
        - ValueError: If the lengths of `initial_thicknesses`, `initial_material_distribution`, 
                      and `initial_incoherency_list` (if provided) are not the same.
        """
        if initial_incoherency_list is not None and len(initial_thicknesses) != len(initial_incoherency_list):
            raise ValueError("Length of initial_thicknesses, and initial_incoherency_list must be the same.")
        
        # Protect internal lists and mappings
        self._thicknesses = initial_thicknesses  # Store layer thicknesses
        self._material_distribution = None  # Store material distribution
        self._fixed_material_distribution = fixed_material_distribution  # Fixed material distribution flag
        self._auto_coherency = auto_coherency  # Auto-coherency flag
        self._is_material_set = False
        
        # Handle fixed material distribution
        if not fixed_material_distribution:
            """
            In kwargs:
            - material_set (List[str], optional): List of material names, required if distribution is not fixed.
            """
            material_set = kwargs.get("material_set", [])
            self._material_set = material_set  # Store material set
            self._is_material_set = True
        
        # Initialize incoherency list based on auto_coherency flag
        if not self._auto_coherency:
            # Use the provided incoherency list
            self._incoherency_list = initial_incoherency_list if initial_incoherency_list is not None else [False] * len(self._thicknesses)
        else:
            # Determine incoherency based on dlist values if auto_coherency is True and no list is provided
            self._incoherency_list = initial_incoherency_list if initial_incoherency_list is not None else self._determine_coherency()
            
        # Enumerate materials
        self._material_enum = self._enumerate_materials(material_set)  # Material to integer mapping

        # Initialize nk functions dictionary using iterate_nk_data
        self._nk_funcs = self._create_nk_funcs(interpolate_nk)  # Initialize nk functions

        # Set initial theta array as None
        self._theta = None

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
        Enumerate materials to create a mapping from integer index to material name.
        
        Args:
        - material_set (List[str]): The set of materials to enumerate.
        
        Returns:
        - Dict[int, str]: A dictionary mapping integer indices to material names.
        """
        # Map material indices to their names
        return {i: material for i, material in enumerate(material_set)}

    def _determine_coherency(self) -> List[bool]:
        """
        Determine the incoherency of layers based on their thickness, wl ...

        Returns:
        - List[bool]: List indicating incoherency (True if incoherent, False if coherent).
        """
        threshold = 300.0  # Threshold in microns
        d_squared = jnp.square(jnp.array(self._thicknesses))  # Compute the square of dlist values
        # Incoherency if the squared thickness exceeds the threshold
        return list(jnp.greater(d_squared, threshold))

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

    @jit
    def _calculate_theta(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the theta array for each layer in the stack.
        This is a placeholder for the actual calculation which might depend on the materials' properties and angles.
        """
        # Placeholder calculation for demonstration purposes
        return jnp.zeros(len(self._thicknesses))  # Replace with actual calculation logic

    # Getter for dlist
    @property
    def dlist(self) -> List[float]:
        """
        Get the list of layer thicknesses.

        Returns:
        - List[float]: List of thicknesses.
        """
        return self._thicknesses

    # Setter for dlist
    @dlist.setter
    def dlist(self, new_thicknesses: List[float]) -> None:
        """
        Set the list of layer thicknesses.

        Args:
        - new_thicknesses (List[float]): New list of thicknesses.
        """
        self._thicknesses = new_thicknesses

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
    def material_distribution(self, new_material_distribution: List[str], theta: jnp.ndarray) -> None:
        """
        Set the material distribution list and update theta.

        Args:
        - new_material_distribution (List[str]): Distribution of materials in the stack.
        - theta (jax.numpy.ndarray): Theta array to be used for calculation.
        """
        # Check if the lengths of the provided lists are consistent
        if len(self._thicknesses) != len(new_material_distribution):
            raise ValueError("Length of initial_thicknesses and new_material_distribution must be the same.")
        if self._fixed_material_distribution and self._is_material_set:
            raise ValueError("Material distribution is fixed and cannot be reassigned.")
        self._material_distribution = new_material_distribution
        self._theta = self._calculate_theta(theta)  # Pass theta to recalculate
        if not self._is_material_set:
            self._material_set = self._scan_material_set(new_material_distribution)
            self._is_material_set = True

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
        self._incoherency_list = new_incoherency_list
    
    # Getter for theta
    @property
    def theta(self) -> jnp.ndarray:
        """
        Get the list of theta jnp array.

        Returns:
        - List[bool]: JAX.numpy array of theta for each layer.
        """
        return self._theta

    def save_log(self, filename: str):
        """
        Save the multilayer stack structure to a log file, including details about each layer and its properties.
    
        Args:
            filename (str): The name of the file to which the log will be saved. This file will contain details about 
                            the multilayer thin film stack, including layer thicknesses, material names, incoherency 
                            flags, theta values, and detailed properties for each layer.
    
        Functionality:
            - Opens the specified file in write mode.
            - Writes a header and summary information about the multilayer stack.
            - Iterates through each layer and writes detailed information including the thickness, refractive index,
              and angles for each layer to the file.
        """
        # Open the specified file in write mode
        with open(filename, 'w') as file:
            # Write a header for the log file
            file.write("Multilayer Thin Film Stack Log\n")
            # Write the total number of layers in the stack
            file.write(f"Total Layers: {len(self._thicknesses)}\n")
            # Write the thicknesses of all layers, formatted as a comma-separated list
            file.write("Layer Thicknesses (nm): " + ', '.join(map(str, self._thicknesses)) + "\n")
            # Write the list of materials used in the stack, formatted as a comma-separated list
            file.write("Materials: " + ', '.join(self._material_set) + "\n")
            # Write the incoherency flags for each layer, formatted as a comma-separated list
            file.write("Incoherency: " + ', '.join(map(str, self._incoherency_list)) + "\n")
            # Write the theta values for each layer, formatted as a comma-separated list
            file.write("Theta values: " + ', '.join(map(str, self._theta)) + "\n")
            # Write a section header for detailed layer properties
            file.write("\nDetailed Layer Properties:\n")
            # Write the column headers for the detailed layer properties table
            file.write("Layer Thickness (nm) - Theta(i)\n")
            
            # Iterate over each layer and write its detailed properties to the file
            for i in range(len(self._thicknesses)):
                # Get the thickness of the current layer
                d = self._thicknesses[i]
                # Get the function to compute the complex refractive index for the current layer
                nk_func = self._nk_funcs[i]
                # Get the angle of incidence/refraction at the current layer interface
                theta_i = self._theta[i]
                # Get the angle at the next layer interface if it exists, otherwise default to 0
                theta_ip1 = self._theta[i+1] if i+1 < len(self._theta) else 0
                # Write the detailed properties of the current layer to the file
                file.write(f"Layer {i + 1} ({self._material_set[i]}): {d} nm, Theta(i): {theta_i}\n")
