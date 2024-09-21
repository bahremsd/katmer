from functools import lru_cache, partial # Importing lru_cache to cache the function results and importing partial decorator
import jax
jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
from jax import jit, device_put, grad, vmap # Import JAX functions for JIT compilation, gradient and vmap
import matplotlib.pyplot as plt # Import matplotlib for plotting
import numpy as np # Importing numpy lib for savetxt function for saving arrays to csv files
import os # Importing os to handle file paths
import pandas as pd # Importing pandas to handle CSV data
from typing import Union, List, Tuple, Optional, Callable # Type hints for function signatures
import warnings # Importing the warnings module to handle warnings in the code

from . import nk_data_dir

@lru_cache(maxsize=32)
def load_nk_data(material_name: str = '') -> Union[jnp.ndarray, None]:
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
    
    @jit  # Just-In-Time compilation using JAX, speeds up the execution by compiling the function once.
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
    nk_data = load_nk_data(material_name)  # Load the nk data for the specified material
    wavelength, refractive_index, extinction_coefficient = nk_data.T  # Transpose to get columns as variables

    # Interpolate refractive index and extinction coefficient
    compute_refractive_index = interpolate_1d(wavelength * 1e-6, refractive_index)  # Convert wavelength to meters for interpolation
    compute_extinction_coefficient = interpolate_1d(wavelength * 1e-6, extinction_coefficient)  # Convert wavelength to meters for interpolation

    @jit  # Just-in-time compile the function to optimize performance
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

    # Ensure the data is on the correct device
    # Move arrays to the appropriate device (e.g., GPU) for processing
    wavelength_arr, refractive_index_arr, extinction_coeff_arr = map(device_put, [wavelength_arr, refractive_index_arr, extinction_coeff_arr])

    # Combine the arrays into a single 2D array
    # Stack arrays as columns into a 2D array for saving
    data = jnp.column_stack((wavelength_arr, refractive_index_arr, extinction_coeff_arr))

    # Construct the file path
    # Create a file path for saving the data based on the material name
    path = os.path.join(nk_data_dir, f'{material_name}.csv')
    
    # Save the file with a header
    # Convert the jax.numpy array to a numpy array for file saving and write to CSV
    np.savetxt(path, np.asarray(data), delimiter=',', header='wavelength_in_um,n,k', comments='')
    
    # Provide feedback on file creation
    # Inform the user whether the file was created or recreated successfully
    print(f"'{os.path.basename(path)}' {'recreated' if os.path.exists(path) else 'created'} successfully.")

def visualize_material_properties(material_name = '', logX = False, logY = False, eV = False, savefig = False, save_path = None):
    # Load the data from the .csv file
    data = np.array(load_nk_data(material_name))
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

def common_wavelength_band(material_list: Optional[List[str]] = None) -> Optional[Tuple[float, float]]:
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
        wavelength_arr = load_nk_data(material_name)[:, 0]
        material_min, material_max = jnp.min(wavelength_arr), jnp.max(wavelength_arr)
        
        # Update the min_wavelength and max_wavelength to find the common range
        min_wavelength = jnp.maximum(min_wavelength, material_min)
        max_wavelength = jnp.minimum(max_wavelength, material_max)
        
        # Early exit if no common range is possible
        if min_wavelength > max_wavelength:
            return None
    
    return min_wavelength, max_wavelength

@partial(jit, static_argnums=(0,))
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
    nk_data = load_nk_data(material_name)

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
    data = load_nk_data(material_name)
    # Unpack the columns: wavelength, refractive index (not used), extinction coefficient
    wavelength, _, k = data.T  # data.T transposes the matrix to easily unpack columns
    # Calculate the absorption coefficient: α(λ) = 4 * π * k / λ
    absorption_coefficient = 4 * jnp.pi * k / wavelength
    # Identify the index of the maximum absorption coefficient
    max_absorption_index = jnp.argmax(absorption_coefficient)

    # Return the wavelength corresponding to the maximum absorption
    return float(wavelength[max_absorption_index])