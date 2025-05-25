import jax.numpy as jnp  # Import the jax numpy module for numerical and mathematical operations, used for efficient array manipulation and computations on CPUs/GPUs/TPUs.
from jax.nn.initializers import glorot_uniform, uniform # Importing the Glorot uniform initializer for stack thicknesses (weights of Stack) and uniform initializer for nk initilization (weights of OneUnkMaterialStack_N|K, OneUnkMaterialSparseStack_N|K) in JAX  
from jax.random import split, normal, choice  # Importing random functions from JAX: PRNGKey for random number generator key, split for splitting keys, normal for normal distribution, and choice for random sampling
from jax import Array  # Import the Array class from jax, which is used for creating arrays in JAX, though it's included here primarily for type specification purpose.
from typing import Union  # Importing type hints: Callable for functions and List for lists of elements  


def thickness_initializer(key: Array, 
                          layer_num: Union[int, Array], 
                          min_thickness: Array, 
                          max_thickness: Array):
    """
    Initializes the thickness of a multilayer thin film stack using the Glorot initializer. 
    This function generates an array of thicknesses for each layer, ensuring that the first thickness 
    values in the "Stack" class are distributed nearly uniform within the specified range 
    [min_thickness, max_thickness]. The initialization is based on the Glorot uniform distribution, 
    which is commonly used to initialize weights in neural networks.

    Arguments:
        - key (jax.Array): A random key used for initializing random numbers in JAX.
        - layer_num (int or jax.Array): The number of layers in the thin film stack.
        - min_thickness (jax.Array): The minimum thickness value for the layers.
        - max_thickness (jax.Array): The maximum thickness value for the layers.

    Return:
        thickness_array (jax.Array): An array of initialized thickness values for each layer, where each value is 
            between `min_thickness` and `max_thickness`.
    """
    
    # Calculate the thickness range (difference between max and min thickness)
    thickness_range = max_thickness - min_thickness
    
    # Initialize the Glorot uniform initializer to generate random values
    initializer = glorot_uniform()
    
    # Use the initializer to generate an array with shape (layer_num, 1) of random values
    # This array will serve as the base for our thickness initialization
    glorot_init_array = initializer(key, (layer_num, 1), jnp.float32)
    
    # Scale the glorot initializer values to lie within the specified thickness range
    # The scaling factor ensures that the initialization is spread across the desired range
    updated_range_array = glorot_init_array * (thickness_range/(2*jnp.sqrt(6/(layer_num+1))))
    
    # Update the thickness values to lie within the range [min_thickness, max_thickness]
    # The + (thickness_range / 2) centers the values around the middle of the range
    # Finally, the min_thickness is added to shift the values to the correct range
    thickness_array = jnp.squeeze(updated_range_array + (thickness_range/2) + min_thickness)
    
    # Return the final array of initialized thickness values
    return thickness_array


def generate_deposition_samples(key: Array, 
                                thicknesses: Array, 
                                deposition_deviation_percentage: Union[float, Array], 
                                sensitivity_optimization_sample_num: Union[int, Array]):
    """
    This function generates a list of deposition thicknesses considering a random deviations from the target thicknesses.
    The deviations are applied within a specified percentage range, and the generated thickness samples are meant for 
    sensitivity optimization in multilayer thin film deposition processes. The function introduces a random deviation 
    in both positive and negative directions for each given thickness value and produces multiple samples based on 
    the specified number (sensitivity_optimization_sample_num).

    Arguments:
        - key (jax.Array): The random key to initialize the random number generation.
        - thicknesses (jax.Array):A 1D array of desired target thickness values for each layer in the thin film.
        - deposition_deviation_percentage (float or jax.Array): The percentage by which the deposition thicknesses can deviate
            from the target (both positive and negative).
        - sensitivity_optimization_sample_num: (int or jax.Array): The number of deposition thickness samples to generate for 
            sensitivity optimization.

    Return:
        - deposition_samples (jax.Array): A 2D array with shape (sensitivity_optimization_sample_num, len(thicknesses)), 
            where each row corresponds to a set of deposition thicknesses considering the random deviations.
    """
    
    # Split the key for subkey generation to ensure random operations are independent
    key, _ = split(key)
    
    # Generate random deviation samples following a normal distribution. Each deviation is for each thickness value.
    deviation_samples = normal(key, (sensitivity_optimization_sample_num, len(thicknesses)))
    
    # Normalize the deviation samples to the range [0, 1]
    normalized_deviation_samples = (deviation_samples - jnp.min(deviation_samples)) / (jnp.max(deviation_samples) - jnp.min(deviation_samples))
    
    # Generate new subkey to ensure randomness for the next operation
    key, _ = split(key)
    
    # Randomly choose whether the deviation is positive (+1) or negative (-1)
    over_or_under_deposition = choice(key, a=jnp.array([-1, 1]), shape=(sensitivity_optimization_sample_num, len(thicknesses)))
    
    # Calculate the ratio matrix by combining normalized deviations with the direction of deviation and the percentage
    ratio_matrix = normalized_deviation_samples * over_or_under_deposition * (1+(deposition_deviation_percentage/100))
    
    # Multiply the ratio matrix by the thickness values to generate the final deposition samples
    deposition_samples = ratio_matrix * thicknesses.T
    
    # Return the final generated deposition samples
    return deposition_samples


def refractive_index_initilizer(key, min_refractive_index, max_refractive_index, num_of_data_points):
    """
    Initializes a set of refractive index values uniformly within a specified range.

    Args:
        key: jax.random.PRNGKey used for random number generation.
        min_refractive_index (float): Minimum possible value of the refractive index.
        max_refractive_index (float): Maximum possible value of the refractive index.
        num_of_data_points (int): Number of data points to generate.

    Returns:
        jnp.ndarray: A 1D JAX array of shape (num_of_data_points,) containing uniformly
                     initialized refractive index values in the range 
                     [min_refractive_index, max_refractive_index).
    """
    refractive_index_scale = max_refractive_index - min_refractive_index
    initializer = uniform(refractive_index_scale)
    uniform_init_value = initializer(key, (num_of_data_points,), jnp.float32) + min_refractive_index
    return uniform_init_value


def extinction_coefficient_initilizer(key, min_extinction_coeff, max_extinction_coeff, num_of_data_points):
    """
    Initializes a set of extinction coefficient values uniformly within a specified range.

    Args:
        key: jax.random.PRNGKey used for random number generation.
        min_extinction_coeff (float): Minimum possible value of the extinction coefficient.
        max_extinction_coeff (float): Maximum possible value of the extinction coefficient.
        num_of_data_points (int): Number of data points to generate.

    Returns:
        jnp.ndarray: A 1D JAX array of shape (num_of_data_points,) containing uniformly
                     initialized extinction coefficient values in the range 
                     [min_extinction_coeff, max_extinction_coeff).
    """
    extinction_coeff_scale = max_extinction_coeff - min_extinction_coeff
    initializer = uniform(extinction_coeff_scale)
    uniform_init_value = initializer(key, (num_of_data_points,), jnp.float32) + min_extinction_coeff
    return uniform_init_value
