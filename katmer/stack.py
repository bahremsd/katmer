import jax.numpy as jnp  # Import the jax numpy module for numerical and mathematical operations, used for efficient array manipulation and computations on CPUs/GPUs/TPUs.
import equinox as eqx  # Importing Equinox, a library for neural networks and differentiable programming in JAX
from jax.nn.initializers import glorot_uniform, uniform # Importing the Glorot uniform initializer for stack thicknesses (weights of Stack) and uniform initializer for nk initilization (weights of OneUnkMaterialStack_N|K, OneUnkMaterialSparseStack_N|K) in JAX  
from jax.random import PRNGKey, split, normal, choice  # Importing random functions from JAX: PRNGKey for random number generator key, split for splitting keys, normal for normal distribution, and choice for random sampling
from jax.lax import fori_loop  # XLA-optimized loop in JAX, without Python's loop overhead, ensuring functional-style.
from jax import Array  # Import the Array class from jax, which is used for creating arrays in JAX, though it's included here primarily for type specification purpose.
from jax.typing import ArrayLike  # Import ArrayLike from jax.typing. It is an annotation for any value that is safe to implicitly cast to a JAX array
from typing import Callable, List, Union, Tuple  # Importing type hints: Callable for functions and List for lists of elements  

from .data import material_distribution_to_set, create_data, repeat_last_element
from .objective import objective_to_function
from .tmm import vectorized_coh_tmm, tmm


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

def determine_coherency(thicknesses: ArrayLike) -> Array:
    """
    Determine the incoherency of layers based on their thickness.

    Args:
    - thicknesses (ArrayLike): Array of thicknesses for the layers.

    Returns:
    - Array: List indicating incoherency (True if incoherent, False if coherent).
    """
    threshold = 1000e-9 
    # Incoherency if the squared thickness exceeds the threshold
    incoherency_list = jnp.greater(thicknesses, threshold)
    int_incoherency_list= incoherency_list.astype(jnp.int32)
    return int_incoherency_list 

def tmm_s_or_p_pol_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization):
    R,T = tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization)
    A = jnp.subtract(1, jnp.add(R,T))
    return jnp.array([R, T, A])

def tmm_u_pol_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list):
    R_s, T_s = tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization = jnp.array([False], dtype=bool))
    R_p, T_p = tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization = jnp.array([True], dtype=bool))
    R = jnp.true_divide(jnp.add(R_s, R_p),2)
    T = jnp.true_divide(jnp.add(T_s, T_p),2)
    A = jnp.subtract(1, jnp.add(R,T))
    return jnp.array([R, T, A])

def tmm_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization):

    result = jnp.select(condlist=[jnp.array_equal(polarization, jnp.array([0], dtype = jnp.int32)),
                                   jnp.array_equal(polarization, jnp.array([1], dtype = jnp.int32)),
                                   jnp.array_equal(polarization, jnp.array([2], dtype = jnp.int32))],
                    choicelist=[tmm_s_or_p_pol_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization =  jnp.array([False], dtype=bool)),
                                tmm_s_or_p_pol_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization =  jnp.array([True], dtype=bool)),
                                tmm_u_pol_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list)])

    return result

def tmm_s_or_p_pol_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization, wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num):
    num_wavelengths, num_angles = wl_angle_shape
    R_T_results = jnp.empty((sensitivity_optimization_sample_num, num_angles, num_wavelengths, 2))
    deposition_samples = generate_deposition_samples(key, thickness_list, deposition_deviation_percentage, sensitivity_optimization_sample_num)

    def one_deviation_tmm(i, R_T_results):
        R, T = tmm(data, material_distribution, deposition_samples.at[i,:].get(), wavelengths, angle_of_incidences, coherency_list, polarization)
        # Update the results array
        R_T_results = R_T_results.at[i,:,:,1].set(R)
        R_T_results = R_T_results.at[i,:,:,2].set(T)
        return R_T_results

    R_T_results = fori_loop(0, sensitivity_optimization_sample_num, one_deviation_tmm, R_T_results)
    R_T_mean = jnp.mean(R_T_results, axis=0)
    R, T = R_T_mean.at[:,:,0].get(), R_T_mean.at[:,:,1].get()
    A = jnp.subtract(1, jnp.add(R,T))
    return jnp.array([R, T, A])

def tmm_u_pol_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num):
    num_wavelengths, num_angles = wl_angle_shape
    R_T_results = jnp.empty((sensitivity_optimization_sample_num, num_angles, num_wavelengths, 2))
    deposition_samples = generate_deposition_samples(key, thickness_list, deposition_deviation_percentage, sensitivity_optimization_sample_num)

    def one_deviation_tmm(i, R_T_results):
        R_s, T_s = tmm(data, material_distribution, deposition_samples.at[i,:].get(), wavelengths, angle_of_incidences, coherency_list, polarization = jnp.array([False], dtype=bool))
        R_p, T_p = tmm(data, material_distribution, deposition_samples.at[i,:].get(), wavelengths, angle_of_incidences, coherency_list, polarization = jnp.array([True], dtype=bool))
        R = jnp.true_divide(jnp.add(R_s, R_p),2)
        T = jnp.true_divide(jnp.add(T_s, T_p),2)
        # Update the results array
        R_T_results = R_T_results.at[i,:,:,1].set(R)
        R_T_results = R_T_results.at[i,:,:,2].set(T)
        return R_T_results

    R_T_results = fori_loop(0, sensitivity_optimization_sample_num, one_deviation_tmm, R_T_results)
    R_T_mean = jnp.mean(R_T_results, axis=0)
    R, T = R_T_mean.at[:,:,0].get(), R_T_mean.at[:,:,1].get()
    A = jnp.subtract(1, jnp.add(R,T))
    return jnp.array([R, T, A])

def tmm_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization, wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num):


    return jnp.select(condlist=[jnp.array_equal(polarization, jnp.array([0], dtype = jnp.int32)),
                                jnp.array_equal(polarization, jnp.array([1], dtype = jnp.int32)),
                                jnp.array_equal(polarization, jnp.array([2], dtype = jnp.int32))],
                    choicelist=[tmm_s_or_p_pol_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, jnp.array([False], dtype=bool), wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num),
                                tmm_s_or_p_pol_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, jnp.array([True], dtype=bool), wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num),
                                tmm_u_pol_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num)])

class Stack(eqx.Module):
    thicknesses: Array
    material_distribution: Array = eqx.static_field()
    data: Array = eqx.static_field()
    wavelength: Array = eqx.static_field()
    angle_of_incidences: Array = eqx.static_field()
    coherency_list: Array = eqx.static_field()
    polarization: Array = eqx.static_field()
    sensitivity_optimization: bool = eqx.static_field()
    deposition_deviation_percentage: Array = eqx.static_field()
    sensitivity_optimization_sample_num: int = eqx.static_field()
    key: Array = eqx.static_field()
    wl_angle_shape: Tuple = eqx.static_field()
    objective_func: Callable = eqx.static_field()
    min_thickness: Array = eqx.static_field()
    max_thickness: Array = eqx.static_field()

    def __init__(self, incoming_medium, outgoing_medium, material_distribution_in_str, light, min_thickness, max_thickness, objective, sensitivity_optimization = False, seed = 1903, deposition_deviation_percentage = 5, sensitivity_optimization_sample_num = 10):
        mediums = [incoming_medium] + material_distribution_in_str + [outgoing_medium]
        material_set, material_distribution = material_distribution_to_set(mediums)
        self.data = create_data(material_set)
        self.key = PRNGKey(seed)
        self.thicknesses = thickness_initializer(self.key, len(material_distribution_in_str), jnp.multiply(min_thickness, 1e6), jnp.multiply(max_thickness, 1e6))
        self.objective_func = objective_to_function(objective)
        self.material_distribution = material_distribution
        self.wavelength = light.wavelength
        self.angle_of_incidences = light.angle_of_incidence
        self.coherency_list = determine_coherency(self.thicknesses)
        self.wl_angle_shape = (len(self.wavelength), len(self.angle_of_incidences))
        self.polarization = light.polarization
        self.sensitivity_optimization = sensitivity_optimization
        self.deposition_deviation_percentage = deposition_deviation_percentage
        self.sensitivity_optimization_sample_num = sensitivity_optimization_sample_num
        self.min_thickness = jnp.multiply(min_thickness, 1e6)
        self.max_thickness = jnp.multiply(max_thickness, 1e6)


    def __call__(self):
        self.coherency_list = determine_coherency(self.thicknesses)
        r_t_a = jnp.select(condlist=[jnp.array_equal(self.sensitivity_optimization, False),
                                jnp.array_equal(self.sensitivity_optimization, True)],
                    choicelist=[tmm_insensitive(data = self.data, material_distribution = self.material_distribution, thickness_list = jnp.multiply(self.thicknesses, 1e-6), wavelengths = self.wavelength, angle_of_incidences = self.angle_of_incidences, coherency_list = self.coherency_list, polarization = self.polarization),
                                tmm_sensitive(data = self.data, material_distribution = self.material_distribution, thickness_list = jnp.multiply(self.thicknesses, 1e-6), wavelengths = self.wavelength, angle_of_incidences = self.angle_of_incidences, coherency_list = self.coherency_list, polarization = self.polarization, wl_angle_shape = self.wl_angle_shape, key = self.key, deposition_deviation_percentage = self.deposition_deviation_percentage, sensitivity_optimization_sample_num = self.sensitivity_optimization_sample_num)])

        objective_result = self.objective_func(r_t_a.at[0].get(), r_t_a.at[1].get(), r_t_a.at[2].get())

        return objective_result

    def update_material_distribution(self, new_material_distribution):
        self.material_distribution = [0] + new_material_distribution + [len(self.data)]

####### Refractive Index Stacks


def refractive_index_initilizer(key, min_refractive_index, max_refractive_index, num_of_data_points):
    refractive_index_scale = max_refractive_index - min_refractive_index
    initializer = uniform(refractive_index_scale)
    uniform_init_value = initializer(key, (num_of_data_points), jnp.float32) + min_refractive_index
    return uniform_init_value

def merge_n_data(fixed_data, refractive_index, dynamic_layer_wavelengths, num_of_data_points):
    dynamic_layer_data = jnp.zeros((3, num_of_data_points))
    dynamic_layer_data = dynamic_layer_data.at[0,:].set(dynamic_layer_wavelengths)
    dynamic_layer_data = dynamic_layer_data.at[1,:].set(refractive_index)
    dynamic_layer_data_expanded = jnp.expand_dims(dynamic_layer_data, axis=0)
    data = jnp.concatenate([fixed_data, dynamic_layer_data_expanded], axis=0)
    return data


def merge_thickness(unknown_layer_thickness, thickness_above_unk, thickness_below_unk):
    thickness_list = thickness_above_unk + [unknown_layer_thickness] + thickness_below_unk
    thickness = jnp.array(thickness_list)
    return thickness

class OneUnkMaterialStack_N(eqx.Module):
    refractive_index: Array
    num_of_data_points: int = eqx.static_field()
    num_of_repeat: int = eqx.static_field()
    max_data_dim: int = eqx.static_field()
    material_distribution: Array = eqx.static_field()
    fixed_data: Array = eqx.static_field()
    thickness_above_unk: Array = eqx.static_field()
    thickness_below_unk: Array = eqx.static_field()
    wavelengths: Array = eqx.static_field()
    dynamic_layer_wavelengths: Array = eqx.static_field()
    angle_of_incidences: Array = eqx.static_field()
    coherency_list: Array = eqx.static_field()
    polarization: Array = eqx.static_field()
    key: Array = eqx.static_field()
    observable_func: Callable = eqx.static_field()
    min_refractive_index: Array = eqx.static_field()
    max_refractive_index: Array = eqx.static_field()

    def __init__(self, incoming_medium, outgoing_medium, material_dist_above_unk_in_str, material_dist_below_unk_in_str, thickness_above_unk, thickness_below_unk, light, coherency_list, min_refractive_index, max_refractive_index, observable, num_of_data_points, seed = 1903):
        num_of_mediums_above_unk = len([incoming_medium] + material_dist_above_unk_in_str)
        mediums = [incoming_medium] + material_dist_above_unk_in_str + material_dist_below_unk_in_str + [outgoing_medium]
        fixed_material_set, fixed_material_distribution = material_distribution_to_set(mediums)
        self.num_of_data_points = num_of_data_points
        self.fixed_data = create_data(fixed_material_set)
        self.thickness_above_unk = thickness_above_unk
        self.thickness_below_unk = thickness_below_unk
        self.key = PRNGKey(seed)
        self.wavelengths = light.wavelength
        self.angle_of_incidences = light.angle_of_incidence
        self.coherency_list = coherency_list
        self.polarization = light.polarization
        self.max_data_dim = self.fixed_data.shape[2]
        self.num_of_repeat = self.max_data_dim - num_of_data_points
        self.refractive_index = refractive_index_initilizer(self.key, min_refractive_index, max_refractive_index, num_of_data_points)
        self.dynamic_layer_wavelengths = jnp.linspace(jnp.min(self.wavelengths), jnp.max(self.wavelengths), num_of_data_points)
        self.dynamic_layer_wavelengths = repeat_last_element(self.dynamic_layer_wavelengths, self.num_of_repeat)
        self.observable_func = objective_to_function(observable)
        self.material_distribution = jnp.insert(fixed_material_distribution, num_of_mediums_above_unk, len(fixed_material_distribution))
        self.min_refractive_index = min_refractive_index
        self.max_refractive_index = max_refractive_index

    def __call__(self, unknown_layer_thickness):

        refractive_index = repeat_last_element(self.refractive_index, self.num_of_repeat)
        data = merge_n_data(self.fixed_data, refractive_index, self.dynamic_layer_wavelengths, self.max_data_dim)
        thickness_list = merge_thickness(unknown_layer_thickness, self.thickness_above_unk, self.thickness_below_unk)
        R,T = tmm_insensitive(data, self.material_distribution, thickness_list, self.wavelengths, self.angle_of_incidences, self.coherency_list, self.polarization)
        A = jnp.subtract(1, jnp.add(R,T)) 
        observable_result = self.observable_func(R,T,A)

        return observable_result


def extinction_coefficient_initilizer(key, min_extinction_coeff, max_extinction_coeff, num_of_data_points):
    extinction_coeff_scale = max_extinction_coeff - min_extinction_coeff
    initializer = uniform(extinction_coeff_scale)
    uniform_init_value = initializer(key, (num_of_data_points), jnp.float32) + min_extinction_coeff
    return uniform_init_value

def merge_nk_data(data_fixed, refractive_index, extinction_coefficient, dynamic_layer_wavelengths, num_of_data_points):
    dynamic_layer_data = jnp.zeros((3, num_of_data_points))
    dynamic_layer_data = dynamic_layer_data.at[0,:].set(dynamic_layer_wavelengths)
    dynamic_layer_data = dynamic_layer_data.at[1,:].set(refractive_index)
    dynamic_layer_data = dynamic_layer_data.at[2,:].set(extinction_coefficient)
    dynamic_layer_data_expanded = jnp.expand_dims(dynamic_layer_data, axis=0)
    data = jnp.concatenate([data_fixed, dynamic_layer_data_expanded], axis=0)
    return data

class OneUnkMaterialStack_NK(eqx.Module):
    refractive_index: Array
    extinction_coefficient: Array
    num_of_data_points: int = eqx.static_field()
    num_of_repeat: int = eqx.static_field()
    max_data_dim: int = eqx.static_field()
    material_distribution: Array = eqx.static_field()
    fixed_data: Array = eqx.static_field()
    thickness_above_unk: Array = eqx.static_field()
    thickness_below_unk: Array = eqx.static_field()
    wavelengths: Array = eqx.static_field()
    dynamic_layer_wavelengths: Array = eqx.static_field()
    angle_of_incidences: Array = eqx.static_field()
    coherency_list: Array = eqx.static_field()
    polarization: Array = eqx.static_field()
    key: Array = eqx.static_field()
    observable_func: Callable = eqx.static_field()
    min_refractive_index: float = eqx.static_field()
    max_refractive_index: float = eqx.static_field()
    min_extinction_coeff: float = eqx.static_field()
    max_extinction_coeff: float = eqx.static_field()

    def __init__(self, incoming_medium, outgoing_medium, material_dist_above_unk_in_str, material_dist_below_unk_in_str, thickness_above_unk, thickness_below_unk, light, coherency_list, min_refractive_index, max_refractive_index, max_extinction_coeff, min_extinction_coeff, observable, num_of_data_points,normalization, seed = 1903):
        num_of_mediums_above_unk = len([incoming_medium] + material_dist_above_unk_in_str)
        mediums = [incoming_medium] + material_dist_above_unk_in_str + material_dist_below_unk_in_str + [outgoing_medium]
        fixed_material_set, fixed_material_distribution = material_distribution_to_set(mediums)
        self.num_of_data_points = num_of_data_points
        self.fixed_data = create_data(fixed_material_set)
        self.thickness_above_unk = thickness_above_unk
        self.thickness_below_unk = thickness_below_unk
        self.key = PRNGKey(seed)
        self.wavelengths = light.wavelength
        self.angle_of_incidences = light.angle_of_incidence
        self.coherency_list = coherency_list
        self.polarization = light.polarization
        self.max_data_dim = self.fixed_data.shape[2]
        self.num_of_repeat = self.max_data_dim - num_of_data_points
        self.refractive_index = refractive_index_initilizer(self.key, min_refractive_index, max_refractive_index, num_of_data_points)
        self.extinction_coefficient = extinction_coefficient_initilizer(self.key, min_extinction_coeff, max_extinction_coeff, num_of_data_points)
        self.dynamic_layer_wavelengths = jnp.linspace(jnp.min(self.wavelengths), jnp.max(self.wavelengths), num_of_data_points)
        self.dynamic_layer_wavelengths = repeat_last_element(self.dynamic_layer_wavelengths, self.num_of_repeat)
        self.observable_func = objective_to_function(observable)
        self.material_distribution = jnp.insert(fixed_material_distribution, num_of_mediums_above_unk, len(fixed_material_distribution))
        self.min_refractive_index = min_refractive_index
        self.max_refractive_index = max_refractive_index
        self.min_extinction_coeff = min_extinction_coeff
        self.max_extinction_coeff = max_extinction_coeff


    def __call__(self, unknown_layer_thickness):

        refractive_index = repeat_last_element(self.refractive_index, self.num_of_repeat)
        extinction_coefficient = repeat_last_element(self.extinction_coefficient, self.num_of_repeat)
        data = merge_nk_data(self.fixed_data, refractive_index, extinction_coefficient, self.dynamic_layer_wavelengths, self.max_data_dim)
        thickness_list = merge_thickness(unknown_layer_thickness, self.thickness_above_unk, self.thickness_below_unk)
        R,T = tmm_insensitive(data, self.material_distribution, thickness_list, self.wavelengths, self.angle_of_incidences, self.coherency_list, self.polarization)
        A = jnp.subtract(1, jnp.add(R,T))
        observable_result = self.observable_func(R,T,A)

        return observable_result
    

####### Refractive Index Sparse Stacks


def coh_tmm_s_or_p_pol(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, polarization):
    R,T = vectorized_coh_tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, polarization)
    A = jnp.subtract(1, jnp.add(R,T))
    return jnp.array([R, T, A])

def coh_tmm_u_pol(data, material_distribution, thickness_list, wavelengths, angle_of_incidences):
    R_s, T_s = vectorized_coh_tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, polarization = jnp.array([False], dtype=bool))
    R_p, T_p = vectorized_coh_tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, polarization = jnp.array([True], dtype=bool))
    R = jnp.true_divide(jnp.add(R_s, R_p),2)
    T = jnp.true_divide(jnp.add(T_s, T_p),2)
    A = jnp.subtract(1, jnp.add(R,T))
    return jnp.array([R, T, A])

def coh_tmm_sparse(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, polarization):

    result = jnp.select(condlist=[jnp.array_equal(polarization, jnp.array([0], dtype = jnp.int32)),
                                   jnp.array_equal(polarization, jnp.array([1], dtype = jnp.int32)),
                                   jnp.array_equal(polarization, jnp.array([2], dtype = jnp.int32))],
                    choicelist=[coh_tmm_s_or_p_pol(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, polarization =  jnp.array([False], dtype=bool)),
                                coh_tmm_s_or_p_pol(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, polarization =  jnp.array([True], dtype=bool)),
                                coh_tmm_u_pol(data, material_distribution, thickness_list, wavelengths, angle_of_incidences)])

    return result


class OneUnkMaterialSparseStack_N(eqx.Module):
    refractive_index: Array
    compression_ratio: Array
    num_of_data_points: int = eqx.static_field()
    num_of_repeat: int = eqx.static_field()
    max_data_dim: int = eqx.static_field()
    material_distribution: Array = eqx.static_field()
    fixed_data: Array = eqx.static_field()
    thickness_above_unk: Array = eqx.static_field()
    thickness_below_unk: Array = eqx.static_field()
    wavelengths: Array = eqx.static_field()
    dynamic_layer_wavelengths: Array = eqx.static_field()
    angle_of_incidences: Array = eqx.static_field()
    polarization: Array = eqx.static_field()
    key: Array = eqx.static_field()
    observable_func: Callable = eqx.static_field()
    min_refractive_index: Array = eqx.static_field()
    max_refractive_index: Array = eqx.static_field()

    def __init__(self, incoming_medium, outgoing_medium, material_dist_above_unk_in_str, material_dist_below_unk_in_str, thickness_above_unk, thickness_below_unk, light, min_refractive_index, max_refractive_index, observable, num_of_data_points, seed = 1903):
        num_of_mediums_above_unk = len([incoming_medium] + material_dist_above_unk_in_str)
        mediums = [incoming_medium] + material_dist_above_unk_in_str + material_dist_below_unk_in_str + [outgoing_medium]
        fixed_material_set, fixed_material_distribution = material_distribution_to_set(mediums)
        self.num_of_data_points = num_of_data_points
        self.fixed_data = create_data(fixed_material_set)
        self.thickness_above_unk = thickness_above_unk
        self.thickness_below_unk = thickness_below_unk
        self.key = PRNGKey(seed)
        self.wavelengths = light.wavelength
        self.angle_of_incidences = light.angle_of_incidence
        self.polarization = light.polarization
        self.max_data_dim = self.fixed_data.shape[2]
        self.num_of_repeat = self.max_data_dim - num_of_data_points
        self.refractive_index = refractive_index_initilizer(self.key, min_refractive_index, max_refractive_index, num_of_data_points)
        self.dynamic_layer_wavelengths = jnp.linspace(jnp.min(self.wavelengths), jnp.max(self.wavelengths), num_of_data_points)
        self.dynamic_layer_wavelengths = repeat_last_element(self.dynamic_layer_wavelengths, self.num_of_repeat)
        self.compression_ratio = jnp.array([1.0])
        self.observable_func = objective_to_function(observable)
        self.material_distribution = jnp.insert(fixed_material_distribution, num_of_mediums_above_unk, len(fixed_material_distribution))
        self.min_refractive_index = min_refractive_index
        self.max_refractive_index = max_refractive_index

    def __call__(self, unknown_layer_thickness):

        unknown_layer_thickness = jnp.multiply(jnp.true_divide(unknown_layer_thickness, 1000), self.compression_ratio)
        refractive_index = repeat_last_element(self.refractive_index, self.num_of_repeat)
        data = merge_n_data(self.fixed_data, refractive_index, self.dynamic_layer_wavelengths, self.max_data_dim)
        thickness_list = merge_thickness(unknown_layer_thickness, self.thickness_above_unk, self.thickness_below_unk)
        R,T = coh_tmm_sparse(data, self.material_distribution, thickness_list, self.wavelengths, self.angle_of_incidences, self.polarization)
        A = jnp.subtract(1, jnp.add(R,T))
        observable_result = self.observable_func(R,T,A)

        return observable_result
    
class OneUnkMaterialSparseStack_NK(eqx.Module):
    refractive_index: Array
    extinction_coefficient: Array
    compression_ratio = Array
    num_of_data_points: int = eqx.static_field()
    num_of_repeat: int = eqx.static_field()
    max_data_dim: int = eqx.static_field()
    material_distribution: Array = eqx.static_field()
    fixed_data: Array = eqx.static_field()
    thickness_above_unk: Array = eqx.static_field()
    thickness_below_unk: Array = eqx.static_field()
    wavelengths: Array = eqx.static_field()
    dynamic_layer_wavelengths: Array = eqx.static_field()
    angle_of_incidences: Array = eqx.static_field()
    polarization: Array = eqx.static_field()
    key: Array = eqx.static_field()
    observable_func: Callable = eqx.static_field()
    min_refractive_index: float = eqx.static_field()
    max_refractive_index: float = eqx.static_field()
    min_extinction_coeff: float = eqx.static_field()
    max_extinction_coeff: float = eqx.static_field()

    def __init__(self, incoming_medium, outgoing_medium, material_dist_above_unk_in_str, material_dist_below_unk_in_str, thickness_above_unk, thickness_below_unk, light, min_refractive_index, max_refractive_index, max_extinction_coeff, min_extinction_coeff, observable, num_of_data_points,normalization, seed = 1903):
        num_of_mediums_above_unk = len([incoming_medium] + material_dist_above_unk_in_str)
        mediums = [incoming_medium] + material_dist_above_unk_in_str + material_dist_below_unk_in_str + [outgoing_medium]
        fixed_material_set, fixed_material_distribution = material_distribution_to_set(mediums)
        self.num_of_data_points = num_of_data_points
        self.fixed_data = create_data(fixed_material_set)
        self.thickness_above_unk = thickness_above_unk
        self.thickness_below_unk = thickness_below_unk
        self.key = PRNGKey(seed)
        self.wavelengths = light.wavelength
        self.angle_of_incidences = light.angle_of_incidence
        self.polarization = light.polarization
        self.max_data_dim = self.fixed_data.shape[2]
        self.num_of_repeat = self.max_data_dim - num_of_data_points
        self.refractive_index = refractive_index_initilizer(self.key, min_refractive_index, max_refractive_index, num_of_data_points)
        self.extinction_coefficient = extinction_coefficient_initilizer(self.key, min_extinction_coeff, max_extinction_coeff, num_of_data_points)
        self.dynamic_layer_wavelengths = jnp.linspace(jnp.min(self.wavelengths), jnp.max(self.wavelengths), num_of_data_points)
        self.dynamic_layer_wavelengths = repeat_last_element(self.dynamic_layer_wavelengths, self.num_of_repeat)
        self.compression_ratio = jnp.array([1.0])
        self.observable_func = objective_to_function(observable)
        self.material_distribution = jnp.insert(fixed_material_distribution, num_of_mediums_above_unk, len(fixed_material_distribution))
        self.min_refractive_index = min_refractive_index
        self.max_refractive_index = max_refractive_index
        self.min_extinction_coeff = min_extinction_coeff
        self.max_extinction_coeff = max_extinction_coeff


    def __call__(self, unknown_layer_thickness):
        
        unknown_layer_thickness = jnp.multiply(jnp.true_divide(unknown_layer_thickness, 1000), self.compression_ratio)
        refractive_index = repeat_last_element(self.refractive_index, self.num_of_repeat)
        extinction_coefficient = repeat_last_element(self.extinction_coefficient, self.num_of_repeat)
        data = merge_nk_data(self.fixed_data, refractive_index, extinction_coefficient, self.dynamic_layer_wavelengths, self.max_data_dim)
        thickness_list = merge_thickness(unknown_layer_thickness, self.thickness_above_unk, self.thickness_below_unk)
        R,T = coh_tmm_sparse(data, self.material_distribution, thickness_list, self.wavelengths, self.angle_of_incidences, self.polarization)
        A = jnp.subtract(1, jnp.add(R,T))
        observable_result = self.observable_func(R,T,A)

        return observable_result