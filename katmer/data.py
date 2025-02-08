import jax.numpy as jnp # Import the jax numpy module for numerical and mathematical operations
from jax import vmap, grad # Import JAX functions for function gradient, vectorization
from jax.lax import fori_loop # This allows us to implement efficient and JIT compiled for-loops
from jax import Array  # Import the Array class from jax for type specification purpose.
from jax.typing import ArrayLike  # Import ArrayLike from jax.typing. It is for safely casting to a JAX array
import matplotlib.pyplot as plt # Import matplotlib for plotting
import numpy as np # Importing numpy lib for savetxt function for saving arrays to csv files
import os # Importing os to handle file paths
import pandas as pd # Importing pandas to handle CSV data
from typing import Union, List, Tuple, Optional, Callable, Dict # Type hints for function signatures
import warnings # Importing the warnings module to handle warnings in the code

from . import nk_data_dir

def load_nk_data_csv(material_name: str = '') -> Union[jnp.ndarray, None]:
    """
    Load the refractive index (n) and extinction coefficient (k) data for a given material: (n + 1j * k).

    This function fetches wavelength-dependent refractive index (n) and extinction coefficient (k)
    data for a specified material. The data is read from a CSV file located in the 'nk_data/' directory.
    The CSV file should be named after the material, e.g., 'Si.csv', and include three columns: wavelength (in micrometers),
    refractive index (n), and extinction coefficient (k). These parameters are crucial for optical simulations,
    allowing the user to work with materials' optical properties over a range of wavelengths.

    Args:
        material_name (str): The name of the material for which the data is to be loaded.
                             This must not be an empty string, and the corresponding CSV file
                             must exist in the 'nk_data/' directory.

    Returns:
        jnp.ndarray: A 2D array containing the wavelength (first column),
                     refractive index (n) (second column), and extinction coefficient (k) (third column).
                     Each row corresponds to a different wavelength.

        None: If the function fails due to any raised exception or if the CSV file is empty,
              it will return None.

    Raises:
        ValueError: If the material name is an empty string.
        FileNotFoundError: If the file for the given material does not exist in the 'nk_data/' folder.
        IOError: If there's an issue reading or parsing the file.
    """
    # Check that the material name is not an empty string
    if not material_name:
        raise ValueError("Material name cannot be an empty string.")  # Raise an error if no material is provided

    # Construct the file path and check if the file exists
    file_path = os.path.join(nk_data_dir, f'{material_name}.csv')  # Create the full path to the file
    if not os.path.exists(file_path):
        # Raise an error if the file for the material does not exist
        raise FileNotFoundError(f"No data found for material '{material_name}' in 'nk_data/' folder (library database).")

    # Load the data from the CSV file
    try:
        # Load the CSV data as a JAX array (important for using JAX's functionality, like automatic differentiation)
        data = jnp.asarray(pd.read_csv(file_path, skiprows=1, header=None).values)
    except Exception as e:
        # If an error occurs during file reading or conversion, raise an IOError
        raise IOError(f"An error occurred while loading data for '{material_name}': {e}")

    # Check if the file is empty or doesn't contain valid data
    if data.size == 0:
        # Raise an error if the data array is empty or incorrectly formatted
        raise ValueError(f"The file for material '{material_name}' is empty or not in the expected format.")

    return data  # Return the loaded data as a JAX array

def load_nk_data_numpy(material_name: str = '') -> Union[jnp.ndarray, None]:
    """
    Load the refractive index (n) and extinction coefficient (k) data for a given material: (n + 1j * k).

    This function fetches wavelength-dependent refractive index (n) and extinction coefficient (k)
    data for a specified material. The data is read from a CSV file located in the 'nk_data/' directory.
    The CSV file should be named after the material, e.g., 'Si.csv', and include three columns: wavelength (in micrometers),
    refractive index (n), and extinction coefficient (k). These parameters are crucial for optical simulations,
    allowing the user to work with materials' optical properties over a range of wavelengths.

    Args:
        material_name (str): The name of the material for which the data is to be loaded.
                             This must not be an empty string, and the corresponding CSV file
                             must exist in the 'nk_data/' directory.

    Returns:
        jnp.ndarray: A 2D array containing the wavelength (first column),
                     refractive index (n) (second column), and extinction coefficient (k) (third column).
                     Each row corresponds to a different wavelength.

        None: If the function fails due to any raised exception or if the CSV file is empty,
              it will return None.

    Raises:
        ValueError: If the material name is an empty string.
        FileNotFoundError: If the file for the given material does not exist in the 'nk_data/' folder.
        IOError: If there's an issue reading or parsing the file.
    """
    # Check that the material name is not an empty string
    if not material_name:
        raise ValueError("Material name cannot be an empty string.")  # Raise an error if no material is provided

    # Construct the file path and check if the file exists
    file_path = os.path.join(nk_data_dir, f'numpy/{material_name}.npy')  # Create the full path to the file
    if not os.path.exists(file_path):
        # Raise an error if the file for the material does not exist
        raise FileNotFoundError(f"No data found for material '{material_name}' in 'nk_data/numpy/' folder (library database).")

    # Load the data from the CSV file
    try:
        # Load the CSV data as a JAX array (important for using JAX's functionality, like automatic differentiation)
        data = jnp.load(file_path)

    except Exception as e:
        # If an error occurs during file reading or conversion, raise an IOError
        raise IOError(f"An error occurred while loading data for '{material_name}': {e}")

    # Check if the file is empty or doesn't contain valid data
    if data.size == 0:
        # Raise an error if the data array is empty or incorrectly formatted
        raise ValueError(f"The file for material '{material_name}' is empty or not in the expected format.")

    return data  # Return the loaded data as a JAX array

def interpolate_1d(x: jnp.ndarray, y: jnp.ndarray) -> Callable[[float], float]:
    """
    Creates a 1D linear interpolation function based on the provided x and y arrays.

    This function returns a callable that performs linear interpolation on the input data points (x, y).
    Given an x value, it finds the corresponding y value by assuming a straight line between two closest points
    in the x array and using the equation of the line.

    Args:
        x (jnp.ndarray): Array of x values (independent variable). It must be sorted in ascending order.
        y (jnp.ndarray): Array of y values (dependent variable). It should have the same length as the x array.

    Returns:
        Callable[[float], float]: A function that, when provided with a single float x value, returns the corresponding
        interpolated float y value based on the linear interpolation.
    """

    def interpolate(x_val: float) -> float:
        # Find the index where x_val would fit in x to maintain the sorted order
        idx = jnp.searchsorted(x, x_val, side='right') - 1
        # Ensure idx is within valid bounds (0 to len(x)-2) to avoid out-of-bounds errors
        idx = jnp.clip(idx, 0, x.shape[0] - 2)

        # Retrieve the two nearest x values, x_i and x_{i+1}, that surround x_val
        x_i, x_ip1 = x[idx], x[idx + 1]
        # Retrieve the corresponding y values, y_i and y_{i+1}, at those x positions
        y_i, y_ip1 = y[idx], y[idx + 1]

        # Calculate the slope of the line between (x_i, y_i) and (x_{i+1}, y_{i+1})
        slope = (y_ip1 - y_i) / (x_ip1 - x_i)

        # Interpolate the y value using the slope formula: y = y_i + slope * (x_val - x_i)
        return y_i + slope * (x_val - x_i)

    return interpolate  # Return the interpolation function to be used later

def interpolate_nk(material_name: str) -> Callable[[float], complex]:
    """
    Load the nk data for a given material and return a callable function that computes
    the complex refractive index for any wavelength.

    Args:
        material_name (str): Name of the material to load the nk data for.

    Returns:
        Callable[[float], complex]: A function that takes a wavelength (in meters) and
                                    returns the complex refractive index.
    """
    nk_data = load_nk_data_numpy(material_name)  # Load the nk data for the specified material
    wavelength, refractive_index, extinction_coefficient = nk_data[0,:], nk_data[1,:], nk_data[2,:]  # Transpose to get columns as variables

    # Interpolate refractive index and extinction coefficient
    compute_refractive_index = interpolate_1d(wavelength * 1e-6, refractive_index)  # Convert wavelength to meters for interpolation
    compute_extinction_coefficient = interpolate_1d(wavelength * 1e-6, extinction_coefficient)  # Convert wavelength to meters for interpolation


    def compute_nk(wavelength: float) -> complex:
        """
        Compute the complex refractive index for a given wavelength.

        Args:
            wavelength (float): Wavelength in meters.

        Returns:
            complex: The complex refractive index, n + i*k, where n is the refractive index
                     and k is the extinction coefficient.
        """
        n = compute_refractive_index(wavelength)  # Get the refractive index at the given wavelength
        k = compute_extinction_coefficient(wavelength)  # Get the extinction coefficient at the given wavelength
        return jnp.array(n + 1j * k)  # Combine n and k into a complex number and return

    return compute_nk  # Return the function that computes the complex refractive index

def repeat_last_element(array: ArrayLike, 
                        repeat: ArrayLike) -> Array:
    """
    The function appends (or pads) the last element of the input `array` to the 
    end of the array `repeat` number of times. It uses the `jnp.pad` function 
    from the JAX library to perform padding, which allows for efficient and 
    differentiable operations on the array.

    Arguments:
    - `array`: An input array-like object that contains numeric or other elements. 
       It is the array on which the operation will be performed.
    - `repeat`: A numeric value that specifies how many times the last element of 
       the `array` should be appended to the end.

    Returns:
    - A new array where the last element of the input `array` is repeated 
      `repeat` number of times and appended to the original array. 
      The returned array is padded with the repeated values of the last element.
    """

    # Padding is applied at the end of the array (0 padding at the start, and `repeat` padding at the end).
    # The padding mode is set to 'constant', meaning a constant value will be used for padding.
    # `array.at[-1].get()` retrieves the last element of the input array to use as the constant padding value.
    padded_array = jnp.pad(array, (0, repeat), mode='constant', constant_values=array.at[-1].get())
    
    # Return the modified array after appending the repeated last element.
    return padded_array

def prepare_material_data(data_container: List, 
                          num_of_materials: int, 
                          max_dim: int) -> Array:
    """
    This function processes a list of material data and standardizes their dimensions 
    by padding or repeating data to match the maximum data length (`max_dim`) among all materials.

    Arguments:
    - data_container: List
        A list containing material data for multiple materials. 
        Each element in this list is a 2D array where:
          - Row 0 represents the wavelength data (`wl`) for the material.
          - Row 1 represents the refractive index (`n`) data for the material.
          - Row 2 represents the extinction coefficient (`k`) data for the material.
        Each material may have a different number of data points.
    
    - num_of_materials: int
        The total number of materials in the `data_container`.
        This determines the number of iterations required to process all materials.

    - max_dim: int
        The maximum length of the data arrays (i.e., the number of data points) 
        across all materials. All materials' data will be padded or repeated to match this length.

    Returns:
    - Array: A 3D JAX array of shape `(num_of_materials, 3, max_dim)` containing:
        - For each material, the first dimension stores the wavelength data (in micrometers).
        - The second dimension stores the refractive index (`n`) data.
        - The third dimension stores the extinction coefficient (`k`) data.
      The function ensures all data arrays are the same size by repeating the last values as necessary.
    """

    # Create an empty 3D array with shape (num_of_materials, 3, max_dim)
    # This array will store the wavelength, n, and k data for all materials after processing.
    data = jnp.empty((num_of_materials, 3, max_dim))
    
    # Loop through each material in the data_container to process its data.
    for i in range(num_of_materials):
        # Retrieve the data for the ith material from the data_container.
        material_data = data_container[i]
        
        # Calculate the number of elements to repeat, which is the difference 
        # between max_dim and the current material's data length.
        repeat = max_dim - (jnp.shape(material_data)[1])
        
        # Repeat the last element of the wavelength data to fill up to max_dim.
        wl_at_i = repeat_last_element(material_data.at[0, :].get(), repeat)
        
        # Repeat the last element of the refractive index (n) data to fill up to max_dim.
        n_at_i = repeat_last_element(material_data.at[1, :].get(), repeat)
        
        # Repeat the last element of the extinction coefficient (k) data to fill up to max_dim.
        k_at_i = repeat_last_element(material_data.at[2, :].get(), repeat)
        
        # Store the processed wavelength data (converted from micrometers to meters).
        data = data.at[i, 0, :].set(jnp.multiply(wl_at_i, 1e-6))
        
        # Store the processed refractive index (n) data into the 3D array.
        data = data.at[i, 1, :].set(n_at_i)
        
        # Store the processed extinction coefficient (k) data into the 3D array.
        data = data.at[i, 2, :].set(k_at_i)
    
    # Return the fully processed 3D array with standardized dimensions for all materials.
    return data

def create_data(material_set: List[str]) -> Array:
    """
    This function loads material data for a given list of materials and prepares a 3D JAX array containing the material properties. 
    The function iterates over the provided material set, loads corresponding data files, and calculates the maximum dimensionality 
    of the data across all materials. It then uses a helper function `prepare_material_data` to organize the material data into 
    a structured format.

    Arguments:
    - `material_set` (List[str]): A list of strings where each string represents the name of a material. These names are used to 
      locate and load the corresponding material data from a predefined directory.

    Returns:
    - A JAX array of shape `(num_of_materials, 3, max_dim)` where:
      - `num_of_materials` is the number of materials in the provided material set.
      - `3` represents three columns of material properties (e.g., wavelength, refractive index, extinction coefficient).
      - `max_dim` is the maximum number of data points among all materials.
    """

    # Initialize the maximum dimension of material data to zero.
    max_dim = 0

    # Create an empty list to store data for each material.
    data_container = []

    # Calculate the total number of materials provided in the material set.
    num_of_materials = len(material_set)

    # Loop through each material in the material set.
    for i in range(num_of_materials):
        # Construct the file path for the current material using its name and the directory structure.
        file_path = os.path.join(nk_data_dir, f'numpy/{material_set[i]}.npy')

        # Load the material data from the .npy file as a JAX array.
        material_data = jnp.load(file_path)

        # Append the loaded material data to the container list.
        data_container.append(material_data)

        # Check if the second dimension of the current material's data exceeds the current max dimension.
        if jnp.shape(material_data)[1] > max_dim:
            # Update the maximum dimension if the current material has more data points.
            max_dim = jnp.shape(material_data)[1]

    # Call a helper function to prepare the material data into a structured JAX array.
    # Pass the list of material data, the number of materials, and the maximum dimension.
    data = prepare_material_data(data_container, num_of_materials, max_dim)

    # Return the prepared 3D JAX array containing all the material data.
    return data

def create_nk_list(material_distribution: ArrayLike, 
                   data: ArrayLike, 
                   wavelength: ArrayLike) -> Array:
    """
    This function generates a list of complex refractive indices (nk) for a multilayer thin film 
    structure based on the provided material distribution, wavelength, and material data. 

    Arguments:
    - material_distribution: A JAX array of integers that indicates the material index for each 
      layer of the multilayer thin film. For example, if `material_distribution[1]` is 1, 
      it implies that the first layer corresponds to the material in the material set at index 1.

    - data: A 3D JAX array containing the wavelength-dependent refractive index (n) and extinction 
      coefficient (k) for all materials in the material set. The array has the shape 
      (num_of_materials, 3, max_dim), where:
        * `num_of_materials` is the total number of materials in the material set.
        * The first dimension (index 0) stores the wavelength values.
        * The second dimension (index 1) stores the refractive indices (n).
        * The third dimension (index 2) stores the extinction coefficients (k).
      
    - wavelength: A scalar that represents wavelength (in meters) at which the refractive index and 
      extinction coefficient will be interpolated.

    Returns:
    - A JAX array of the same length as `material_distribution`, where each element represents 
      the complex refractive index (n + i*k) of the corresponding layer at the specified wavelength.
    """

    def update_nk_list(i, nk_carry):
        # Retrieve the index of the material for the i-th layer from material_distribution
        material_data_idx = material_distribution.at[i].get()

        # Interpolate the refractive index (n) for the i-th layer at the given wavelength
        n_at_i = jnp.interp(wavelength, 
                            data.at[material_data_idx, 0, :].get(), 
                            data.at[material_data_idx, 1, :].get())
        
        # Interpolate the extinction coefficient (k) for the i-th layer at the given wavelength
        k_at_i = jnp.interp(wavelength, 
                            data.at[material_data_idx, 0, :].get(), 
                            data.at[material_data_idx, 2, :].get())
        
        # Compute the complex refractive index (nk = n + i*k) for the i-th layer
        nk_at_i = jnp.add(n_at_i, jnp.multiply(1j, k_at_i))
        
        # Update the nk_carry array with the computed nk value for the i-th layer
        nk_carry = nk_carry.at[i].set(nk_at_i)
        
        # Return the updated nk_carry array
        return nk_carry

    # Initialize the nk_carry array as a complex64 array with the same size as material_distribution
    # Use the fori_loop to iterate over all layers and populate the nk_carry array with nk values
    complete_nk_list = fori_loop(0, 
                                 len(material_distribution), 
                                 update_nk_list, 
                                 material_distribution.astype('complex64'))
    
    # Return the complete list of complex refractive indices for all layers
    return complete_nk_list

def material_distribution_to_set(material_distribution: List[str]) -> Tuple:
    """
    This function takes a list of material names representing a multilayer thin film 
    and converts it into two outputs: a unique set of materials and a corresponding list 
    of integers that represent the materials by their enumerated indices. This is useful 
    for simplifying the representation of material sequences in numerical simulations 
    or computations. For example, a material sequence `["SiO2", "TiO2", "Al2O3", "TiO2", 
    "Al2O3", "SiO2"]` will be converted into a material set `["SiO2", "TiO2", "Al2O3"]` 
    and an enumerated list `[0, 1, 2, 1, 2, 0]`.

    Arguments:
    material_distribution: List[str]
        A list of strings where each string represents the name of a material 
        in a multilayer thin film. The list may contain duplicate entries, as it 
        reflects the sequence of materials in the structure.

    Returns:
    Tuple[List[str], jax.numpy.ndarray]
        - A list of unique material names, representing all distinct materials used 
          in the input `material_distribution`.
        - A JAX NumPy array of integers where each integer corresponds to the 
          index of a material in the unique material set. The order of these integers 
          reflects the order of materials in the input `material_distribution`.
    """

    # Create a unique list of materials by converting the input list to a set (removing duplicates)
    # and then back to a list to preserve order.
    material_set = list(set(material_distribution))

    # Create a dictionary where each material in the set is assigned a unique integer index.
    # This will allow mapping from material names to integers.
    material_enum = {material: i for i, material in enumerate(material_set)}

    # Map each material in the input list to its corresponding integer index in `material_enum`.
    # The result is a list of integers that mirrors the input material distribution in order.
    material_list = jnp.array([int(material_enum[material]) for material in material_distribution])

    # Return the unique material set and the corresponding enumerated material list.
    return material_set, material_list

def add_material_to_nk_database(wavelength_arr, refractive_index_arr, extinction_coeff_arr, material_name=''):
    """
    Add material properties to the nk database by saving the data into a CSV file.

    This function validates and saves material properties such as wavelength, refractive index,
    and extinction coefficient into a CSV file. The file is named based on the provided material name.

    Args:
        wavelength_arr (jnp.ndarray): Array of wavelengths in micrometers.
        refractive_index_arr (jnp.ndarray): Array of refractive indices corresponding to the wavelengths.
        extinction_coeff_arr (jnp.ndarray): Array of extinction coefficients corresponding to the wavelengths.
        material_name (str): The name of the material, which is used to name the output CSV file.

    Raises:
        TypeError: If any of the input arrays are not of type jax.numpy.ndarray.
        ValueError: If the input arrays have different lengths or if the material name is empty.
    """

    # Validate input types
    # Check if all input arrays are of type jax.numpy.ndarray
    if not all(isinstance(arr, jnp.ndarray) for arr in [wavelength_arr, refractive_index_arr, extinction_coeff_arr]):
        raise TypeError("All input arrays must be of type jax.numpy.ndarray")

    # Ensure all arrays have the same length
    # Check if the length of refractive_index_arr and extinction_coeff_arr match wavelength_arr
    if not all(len(arr) == len(wavelength_arr) for arr in [refractive_index_arr, extinction_coeff_arr]):
        raise ValueError("All input arrays must have the same length")

    # Validate material name
    # Ensure that the material name is not an empty string
    if not material_name.strip():
        raise ValueError("Material name cannot be an empty string")

    # Check for extinction coefficients greater than 20
    # Warn and threshold extinction coefficients greater than 20 to 20
    if jnp.any(extinction_coeff_arr > 20):
        warnings.warn("Extinction coefficient being greater than 20 indicates that the material is almost opaque. "
                      "In the Transfer Matrix Method, to avoid the coefficients going to 0 and the gradient being zero, "
                      "extinction coefficients greater than 20 have been thresholded to 20.", UserWarning)
        extinction_coeff_arr = jnp.where(extinction_coeff_arr > 20, 20, extinction_coeff_arr)

    # Combine the arrays into a single 2D array
    # Stack arrays as columns into a 2D array for saving
    data = jnp.column_stack((wavelength_arr, refractive_index_arr, extinction_coeff_arr))

    # Construct the file path
    # Create a file path for saving the data based on the material name
    path_csv = os.path.join(nk_data_dir, f'csv/{material_name}.csv')
    path_npy = os.path.join(nk_data_dir, f'numpy/{material_name}.npy')

    # Save the file with a header
    # Convert the jax.numpy array to a numpy array for file saving and write to CSV
    np.savetxt(path_csv, np.asarray(data), delimiter=',', header='wavelength_in_um,n,k', comments='')
    np.save(path_npy, np.asarray(data))

    # Provide feedback on file creation
    # Inform the user whether the file was created or recreated successfully
    print(f"'{os.path.basename(path_csv)}' {'recreated' if os.path.exists(path_csv) else 'created'} successfully.")


def visualize_material_properties(material_name = '', logX = False, logY = False, eV = False, savefig = False, save_path = None):
    # Load the data from the .csv file
    data = np.array(load_nk_data_csv(material_name))
    # Unpack the columns: wavelength, refractive index, extinction coefficient
    wavelength, refractive_index, extinction_coeff = data.T  # wavelength is in um
    # Custom chart specs
    if eV:
        eV_arr = 1239.8/(wavelength*1e3) # E(eV) = 1239.8 / wavelength (nm)
    # Creating plot for refractive_index
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_n = 'navy'
    ax1.set_ylabel('Refractive Index (n)', color=color_n, fontsize=14, fontweight='bold')
    if not eV:
        ax1.set_xlabel('Wavelength (um)', fontsize=14, fontweight='bold')
        ax1.plot(wavelength, refractive_index, color=color_n, linewidth=2, label='Refractive Index (n)')
    else:
        ax1.set_xlabel('Photon energy (eV)', fontsize=14, fontweight='bold')
        ax1.plot(eV_arr, refractive_index, color=color_n, linewidth=2, label='Refractive Index (n)')
    ax1.tick_params(axis='y', labelcolor=color_n, labelsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    # Creating a second y-axis for the extinction coefficient (k)
    ax2 = ax1.twinx()
    color_k = 'crimson'
    ax2.set_ylabel('Extinction Coefficient (k)', color=color_k, fontsize=14, fontweight='bold')
    if not eV:
        ax2.plot(wavelength, extinction_coeff, color=color_k, linewidth=2, linestyle='-', label='Extinction Coefficient (k)')
    else:
        ax2.plot(eV_arr, extinction_coeff, color=color_k, linewidth=2, linestyle='-', label='Extinction Coefficient (k)')
    ax2.tick_params(axis='y', labelcolor=color_k, labelsize=12)
    if logX:
        # Set the x-axis to logarithmic scale
        plt.xscale('log')
    if logY:
        # Set the y-axis to logarithmic scale
        plt.yscale('log')
    # Adding title
    plt.title(f'Refractive Index (n) and Extinction Coefficient (k) vs Wavelength for {material_name}', fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout()
    # Save the figure as a PNG if savefig True
    if savefig:
        # Check that save_path is not an empty string or None
        if not save_path:
            raise ValueError("save_path cannot be an empty string or None")
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)
        # Construct the full save path with filename
        full_save_path = os.path.join(save_path, f'{material_name}_nk_plot.png')
        # Save the figure
        plt.savefig(full_save_path, dpi=300)
        print(f"Figure saved successfully at: {full_save_path}")
    plt.show()

def common_wavelength_band(material_list: List[str]) -> Tuple[float, float]:
    """
    Compute the common wavelength band across a list of materials based on their n-k data.

    Args:
    ----------
    material_list : Optional[List[str]]
        A list of material names for which the common wavelength band is to be calculated.

    Returns:
    -------
    Optional[Tuple[float, float]]
        A tuple containing the minimum and maximum wavelength of the common band.
        Returns None if no common wavelength band exists.

    Raises:
    ------
    ValueError:
        If the material_list is empty or None.
    """
    if not material_list:
        raise ValueError("Material list cannot be empty or None.")

    # Initialize wavelength bounds
    min_wavelength = -jnp.inf
    max_wavelength = jnp.inf

    # Iterate through each material's wavelength range
    for material_name in material_list:
        wavelength_arr = load_nk_data_csv(material_name)[:, 0]
        material_min, material_max = jnp.min(wavelength_arr), jnp.max(wavelength_arr)

        # Update the min_wavelength and max_wavelength to find the common range
        min_wavelength = jnp.maximum(min_wavelength, material_min)
        max_wavelength = jnp.minimum(max_wavelength, material_max)

        # Early exit if no common range is possible
        if min_wavelength > max_wavelength:
            return None

    return min_wavelength, max_wavelength


def calculate_chromatic_dispersion(material_name: str) -> jnp.ndarray:
    """
    Calculate the chromatic dispersion, which is the derivative of the refractive index
    with respect to wavelength.

    Args:
        material_name (str): Name of the material.

    Returns:
        jnp.ndarray: Array containing the chromatic dispersion (d n / d wavelength).
    """
    # Fetch the nk data for the material
    nk_data = load_nk_data_csv(material_name)

    # Unpack the columns: wavelength, refractive index, extinction coefficient
    wavelength, refractive_index, _ = nk_data.T  # nk_data.T transposes the matrix to easily unpack columns

    # Define a function to compute the refractive index as a function of wavelength
    def n_func(wl: jnp.ndarray) -> jnp.ndarray:
        return jnp.interp(wl, wavelength, refractive_index)

    # Compute the derivative of the refractive index function with respect to wavelength
    dn_dw = vmap(grad(n_func))(wavelength)

    return dn_dw

def get_max_absorption_wavelength(material_name: str) -> float:
    """
    Calculate the wavelength at which the absorption coefficient is maximized.

    Args:
        material_name (str): Name of the material.

    Returns:
        float: Wavelength (in μm) corresponding to the maximum absorption coefficient.
    """
    # Fetch the nk data for the material
    data = load_nk_data_csv(material_name)
    # Unpack the columns: wavelength, refractive index (not used), extinction coefficient
    wavelength, _, k = data.T  # data.T transposes the matrix to easily unpack columns
    # Calculate the absorption coefficient: α(λ) = 4 * π * k / λ
    absorption_coefficient = 4 * jnp.pi * k / wavelength
    # Identify the index of the maximum absorption coefficient
    max_absorption_index = jnp.argmax(absorption_coefficient)

    # Return the wavelength corresponding to the maximum absorption
    return float(wavelength[max_absorption_index])