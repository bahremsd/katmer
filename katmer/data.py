from functools import lru_cache, partial
import jax.numpy as jnp
from jax import jit, device_put, grad, vmap
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Union, List, Tuple, Optional, Callable
import warnings

@lru_cache(maxsize=32)
def load_nk_data(material_name: str = '') -> Union[jnp.ndarray, None]:
    """
    Load the refractive index (n) and extinction coefficient (k) data for a given material.

    This function loads data from a CSV file located in the 'nk_data/' directory.
    The file should be named after the material, e.g., 'Si.csv', and contain 
    wavelength, n, and k values.

    Args:
        material_name (str): The name of the material. Must not be an empty string.

    Returns:
        jnp.ndarray: A 2D array containing the wavelength, n, and k values.

    Raises:
        ValueError: If the material name is an empty string.
        FileNotFoundError: If the corresponding material file does not exist.
        IOError: If there is an issue reading the file.
    """
    # Check that the material name is not an empty string
    if not material_name:
        raise ValueError("Material name cannot be an empty string.")
    # Construct the file path and check if the file exists
    file_path = os.getcwd()+"/katmer/" + os.path.join('nk_data', f'{material_name}.csv') #delete this os.getcwd()+"/katmer/" after setup.py installation
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data found for material '{material_name}' in 'nk_data/' folder (library database).")
    # Load the data from the .csv file
    try:
        data = jnp.asarray(pd.read_csv(file_path, skiprows=1, header=None).values)
    except Exception as e:
        raise IOError(f"An error occurred while loading data for '{material_name}': {e}")
    if data.size == 0:
        raise ValueError(f"The file for material '{material_name}' is empty or not in the expected format.")
    
    return data

def interpolate_1d(x: jnp.ndarray, y: jnp.ndarray) -> Callable[[float], float]:
    """
    Creates a function for linear interpolation based on the provided x and y arrays.

    Args:
        x (jnp.ndarray): Array of x values (independent variable).
        y (jnp.ndarray): Array of y values (dependent variable).

    Returns:
        Callable[[float], float]: A function that takes a single x value and returns the corresponding interpolated y value.
    """
    @jit
    def interpolate(x_val: float) -> float:
        idx = jnp.searchsorted(x, x_val, side='right') - 1
        idx = jnp.clip(idx, 0, x.shape[0] - 2)
        
        x_i, x_ip1 = x[idx], x[idx + 1]
        y_i, y_ip1 = y[idx], y[idx + 1]
        
        slope = (y_ip1 - y_i) / (x_ip1 - x_i)
        return y_i + slope * (x_val - x_i)
    
    return interpolate

def interpolate_nk(material_name: str) -> Callable[[float], complex]:
    """
    Load the nk data for a given material and return a callable function that computes
    the complex refractive index for any wavelength.

    Args:
        material_name (str): Name of the material to load the nk data for.

    Returns:
        Callable[[float], complex]: A function that takes a wavelength and returns the complex refractive index.
    """
    nk_data = load_nk_data(material_name)
    wavelength, refractive_index, extinction_coefficient = nk_data.T  # wavelength is in um
    compute_refractive_index = interpolate_1d(wavelength*1e-6, refractive_index) # wavelength is in m
    compute_extinction_coefficient = interpolate_1d(wavelength*1e-6, extinction_coefficient) # wavelength is in m
    
    @jit
    def compute_nk(wavelength: float) -> complex:
        n = compute_refractive_index(wavelength)
        k = compute_extinction_coefficient(wavelength)
        return n + 1j * k
    
    return compute_nk

def add_material_to_nk_database(wavelength_arr, refractive_index_arr, extinction_coeff_arr, material_name=''):
    """
    Add material properties to the nk database by saving the data into a CSV file.
    
    Args:
        wavelength_arr (jnp.ndarray): Array of wavelengths.
        refractive_index_arr (jnp.ndarray): Array of refractive indices.
        extinction_coeff_arr (jnp.ndarray): Array of extinction coefficients.
        material_name (str): The name of the material (used for the filename).
    
    Raises:
        TypeError: If the input arrays are not jax.numpy arrays.
        ValueError: If input arrays have different lengths or if material name is empty.
    """
    # Validate input types
    if not all(isinstance(arr, jnp.ndarray) for arr in [wavelength_arr, refractive_index_arr, extinction_coeff_arr]):
        raise TypeError("All input arrays must be of type jax.numpy.ndarray")

    # Ensure all arrays have the same length
    if not all(len(arr) == len(wavelength_arr) for arr in [refractive_index_arr, extinction_coeff_arr]):
        raise ValueError("All input arrays must have the same length")

    # Validate material name
    if not material_name.strip():
        raise ValueError("Material name cannot be an empty string")

    # Check for extinction coefficients greater than 20
    if jnp.any(extinction_coeff_arr > 20):
        warnings.warn("Extinction coefficient being greater than 20 indicates that the material is almost opaque. "
                      "In the Transfer Matrix Method, to avoid the coefficients going to 0 and the gradient being zero, "
                      "extinction coefficients greater than 20 have been thresholded to 20.", UserWarning)
        extinction_coeff_arr = jnp.where(extinction_coeff_arr > 20, 20, extinction_coeff_arr)

    # Ensure the data is on the correct device
    wavelength_arr, refractive_index_arr, extinction_coeff_arr = map(device_put, [wavelength_arr, refractive_index_arr, extinction_coeff_arr])

    # Combine the arrays into a single 2D array
    data = jnp.column_stack((wavelength_arr, refractive_index_arr, extinction_coeff_arr))

    # Construct the file path
    path = os.path.join('nk_data', f'{material_name}.csv')
    # Save the file with a header
    np.savetxt(path, np.asarray(data), delimiter=',', header='wavelength_in_um,n,k', comments='')
    # Provide feedback on file creation
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