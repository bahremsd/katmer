import jax.numpy as jnp  # Efficient numerical operations on CPU/GPU/TPU using JAX
from jax import Array  # JAX-specific array object for type annotations.
from jax.typing import ArrayLike  # Type hint for anything convertible to a JAX array


def determine_coherency(thicknesses: ArrayLike) -> Array:
    """
    Determines which layers in a multilayer structure are considered incoherent
    based on their physical thickness.

    Incoherent layers typically exceed a certain optical path length and 
    therefore interfere less predictably. This function flags such layers.

    Args:
        thicknesses (ArrayLike): 1D array of layer thicknesses in meters.

    Returns:
        Array: An array of integers (0 or 1), where 1 indicates the corresponding
               layer is incoherent (thickness > 1000 nm), and 0 indicates coherence.
    """
    threshold = 1000e-9  # Coherency threshold in meters (1000 nm)
    incoherency_list = jnp.greater(thicknesses, threshold)  # Boolean mask
    int_incoherency_list = incoherency_list.astype(jnp.int32)  # Convert to int (0 or 1)
    return int_incoherency_list


def merge_thickness(unknown_layer_thickness, thickness_above_unk, thickness_below_unk):
    """
    Combines the thickness of the unknown layer with the known thicknesses of
    the layers above and below it to form a complete stack.

    Args:
        unknown_layer_thickness (float or Array): Thickness of the unknown layer.
        thickness_above_unk (list or Array): Thicknesses of layers above the unknown layer.
        thickness_below_unk (list or Array): Thicknesses of layers below the unknown layer.

    Returns:
        Array: A 1D array of combined thickness values.
    """
    # Concatenate all layers in proper order
    thickness_list = thickness_above_unk + [unknown_layer_thickness] + thickness_below_unk
    thickness = jnp.squeeze(jnp.array(thickness_list))  # Convert to JAX array and remove single dimensions
    return thickness


def merge_n_data(fixed_data, refractive_index, dynamic_layer_wavelengths, num_of_data_points):
    """
    Combines fixed n (real part of refractive index) data with dynamic layer n data.

    Args:
        fixed_data (Array): The existing fixed n data for other layers, shape (N_layers, 3, N_points).
        refractive_index (Array): n values for the unknown dynamic layer, shape (N_points,).
        dynamic_layer_wavelengths (Array): Wavelengths corresponding to the n values, shape (N_points,).
        num_of_data_points (int): Number of wavelength data points.

    Returns:
        Array: Updated data array including the dynamic layer's refractive index.
               Shape becomes (N_layers + 1, 3, N_points).
    """
    # Create a zero-initialized data block for the dynamic layer
    dynamic_layer_data = jnp.zeros((3, num_of_data_points))
    # Fill in wavelength and refractive index data
    dynamic_layer_data = dynamic_layer_data.at[0, :].set(dynamic_layer_wavelengths)
    dynamic_layer_data = dynamic_layer_data.at[1, :].set(refractive_index)
    # Expand to match dimensions for concatenation
    dynamic_layer_data_expanded = jnp.expand_dims(dynamic_layer_data, axis=0)
    # Concatenate along the layer axis
    data = jnp.concatenate([fixed_data, dynamic_layer_data_expanded], axis=0)
    return data


def merge_nk_data(data_fixed, refractive_index, extinction_coefficient, dynamic_layer_wavelengths, num_of_data_points):
    """
    Combines fixed nk (complex refractive index) data with dynamic layer nk data.

    Args:
        data_fixed (Array): The existing fixed nk data for other layers, shape (N_layers, 3, N_points).
        refractive_index (Array): n values for the unknown dynamic layer, shape (N_points,).
        extinction_coefficient (Array): k values for the unknown dynamic layer, shape (N_points,).
        dynamic_layer_wavelengths (Array): Wavelengths corresponding to n and k values, shape (N_points,).
        num_of_data_points (int): Number of wavelength data points.

    Returns:
        Array: Updated data array including the dynamic layer's nk values.
               Shape becomes (N_layers + 1, 3, N_points).
    """
    # Allocate new data block for the dynamic layer
    dynamic_layer_data = jnp.zeros((3, num_of_data_points))
    # Fill in wavelength, n, and k values
    dynamic_layer_data = dynamic_layer_data.at[0, :].set(dynamic_layer_wavelengths)
    dynamic_layer_data = dynamic_layer_data.at[1, :].set(refractive_index)
    dynamic_layer_data = dynamic_layer_data.at[2, :].set(extinction_coefficient)
    # Reshape to add it as a new layer
    dynamic_layer_data_expanded = jnp.expand_dims(dynamic_layer_data, axis=0)
    # Concatenate with existing data
    data = jnp.concatenate([data_fixed, dynamic_layer_data_expanded], axis=0)
    return data
