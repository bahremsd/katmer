import jax.numpy as jnp  # JAX's NumPy module for GPU/TPU-accelerated array computations.
import equinox as eqx  # Equinox is a neural network/differentiable programming library for JAX.
from jax.random import PRNGKey  # PRNGKey is used for random number generation in JAX.
from jax import Array  # Type hint for JAX arrays.
from typing import Callable, Tuple  # Typing support for clarity.

# Utility and model imports specific to thin-film design and modeling.
from tmmax.data import material_distribution_to_set, create_data, repeat_last_element
from .initializers import (
    thickness_initializer,
    refractive_index_initilizer,
    extinction_coefficient_initilizer,
)
from .objective import objective_to_function
from .nk_post_processors import smooth_nk
from .tmm import tmm_insensitive, tmm_sensitive, coh_tmm_sparse
from .utils import (
    determine_coherency,
    merge_thickness,
    merge_n_data,
    merge_nk_data,
)

class Stack(eqx.Module):
    """
    Equinox model representing a multilayer optical thin film stack for optimizing
    layer thicknesses under an objective function using gradient-based methods.

    Attributes:
        thicknesses (Array): Layer thicknesses (in microns).
        material_distribution (Array): Encoded list of material indices per layer.
        data (Array): Optical data (n, k) for each material in the stack.
        wavelength (Array): Wavelengths (in microns) used for simulation.
        angle_of_incidences (Array): Angles (in degrees) for simulation.
        coherency_list (Array): Determines coherent/incoherent modeling per layer.
        polarization (Array): Polarization mode(s) of the incident light.
        sensitivity_optimization (bool): Whether to simulate fabrication variation.
        deposition_deviation_percentage (Array): Deviation % for thickness during sensitivity.
        sensitivity_optimization_sample_num (int): Sample count for averaging in sensitivity analysis.
        key (Array): JAX PRNGKey for reproducibility.
        wl_angle_shape (Tuple): Shape = (num wavelengths, num angles), for reshaping outputs.
        objective_func (Callable): Objective function to optimize (e.g., maximize transmission).
        min_thickness (Array): Minimum allowed thickness for optimization (in microns).
        max_thickness (Array): Maximum allowed thickness for optimization (in microns).
    """

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

    def __init__(
        self,
        incoming_medium,
        outgoing_medium,
        material_distribution_in_str,
        light,
        min_thickness,
        max_thickness,
        objective,
        sensitivity_optimization=False,
        seed=1903,
        deposition_deviation_percentage=5,
        sensitivity_optimization_sample_num=10
    ):
        """
        Initializes the Stack class with simulation parameters.

        Args:
            incoming_medium (str): Name of the input medium (e.g., 'air').
            outgoing_medium (str): Name of the output medium (e.g., 'glass').
            material_distribution_in_str (List[str]): List of materials in stack (excluding input/output).
            light: Light configuration object with attributes:
                - wavelength (Array): Wavelengths in microns.
                - angle_of_incidence (Array): Angles in degrees.
                - polarization (Array): Polarization type(s).
            min_thickness (float): Minimum thickness (in microns) for each layer.
            max_thickness (float): Maximum thickness (in microns) for each layer.
            objective (str): Name of the objective function to use (e.g., 'maximize_transmission').
            sensitivity_optimization (bool): Enables variation-based robustness optimization.
            seed (int): Random seed for reproducibility.
            deposition_deviation_percentage (float): Thickness noise percentage for robustness.
            sensitivity_optimization_sample_num (int): Number of samples for robustness averaging.
        """
        # Combine input, layers, and output to build the full material sequence
        mediums = [incoming_medium] + material_distribution_in_str + [outgoing_medium]

        # Convert to material index set and ordered distribution
        material_set, material_distribution = material_distribution_to_set(mediums)

        # Fetch refractive index and extinction coefficient data for each material
        self.data = create_data(material_set)

        # Initialize JAX random number generator key
        self.key = PRNGKey(seed)

        # Randomly initialize thicknesses within the given bounds (converted to nanometers)
        self.thicknesses = thickness_initializer(
            self.key,
            len(material_distribution_in_str),
            jnp.multiply(min_thickness, 1e6),
            jnp.multiply(max_thickness, 1e6)
        )

        # Convert string/objective to actual callable function
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
        """
        Evaluates the objective function for the current stack configuration.

        Returns:
            float: The scalar result of the objective function (e.g., error or efficiency).
        """
        # Update coherency list based on current thickness values
        self.coherency_list = determine_coherency(self.thicknesses)

        # Choose between regular TMM and sensitivity-aware TMM
        r_t_a = jnp.select(
            condlist=[
                jnp.array_equal(self.sensitivity_optimization, False),
                jnp.array_equal(self.sensitivity_optimization, True)
            ],
            choicelist=[
                # TMM without considering deposition variation
                tmm_insensitive(
                    data=self.data,
                    material_distribution=self.material_distribution,
                    thickness_list=jnp.multiply(self.thicknesses, 1e-6),
                    wavelengths=self.wavelength,
                    angle_of_incidences=self.angle_of_incidences,
                    coherency_list=self.coherency_list,
                    polarization=self.polarization,
                ),
                # TMM with deposition sensitivity sampling
                tmm_sensitive(
                    data=self.data,
                    material_distribution=self.material_distribution,
                    thickness_list=jnp.multiply(self.thicknesses, 1e-6),
                    wavelengths=self.wavelength,
                    angle_of_incidences=self.angle_of_incidences,
                    coherency_list=self.coherency_list,
                    polarization=self.polarization,
                    wl_angle_shape=self.wl_angle_shape,
                    key=self.key,
                    deposition_deviation_percentage=self.deposition_deviation_percentage,
                    sensitivity_optimization_sample_num=self.sensitivity_optimization_sample_num,
                )
            ]
        )

        # Apply the objective function to the simulated reflectance, transmittance, and absorptance
        objective_result = self.objective_func(
            r_t_a.at[0].get(),  # Reflectance
            r_t_a.at[1].get(),  # Transmittance
            r_t_a.at[2].get()   # Absorptance
        )

        return objective_result

    def update_material_distribution(self, new_material_distribution):
        """
        Updates the material distribution for the internal thin-film layers.

        Args:
            new_material_distribution (List[int]): A new list of material indices for the layers.
        """
        # Always keep the first and last indices as incoming/outgoing media
        self.material_distribution = [0] + new_material_distribution + [len(self.data)]


class OneUnkMaterialStack_N(eqx.Module):
    """
    Equinox module representing a multilayer thin film stack model
    with a single unknown refractive index layer. This is used for
    gradient-based optimization of refractive index `n` using experimental data.
    """

    # Trainable parameter: the unknown refractive index of the target material layer
    refractive_index: Array

    # === Static fields for constant parameters ===
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
    smoothing_method: str = eqx.static_field()
    smoothing_window_range: Tuple[float, float] = eqx.static_field()
    smoothing_sigma: float = eqx.static_field()
    smoothing_amplitude: float = eqx.static_field()
    smoothing_kernel_size: int = eqx.static_field()
    smoothing_decay: float = eqx.static_field()

    def __init__(self, 
                 incoming_medium,
                 outgoing_medium,
                 material_dist_above_unk_in_str,
                 material_dist_below_unk_in_str,
                 thickness_above_unk,
                 thickness_below_unk,
                 light,
                 coherency_list,
                 min_refractive_index,
                 max_refractive_index,
                 observable,
                 smoothing_method: str = "gaussian",
                 smoothing_window_range: Tuple[float, float] = (-10.0, 10.0),
                 smoothing_sigma: float = 4.0,
                 smoothing_amplitude: float = 0.5,
                 smoothing_kernel_size: int = 20,
                 smoothing_decay: float = 0.8,
                 num_of_data_points: int = 350,
                 num_of_data_points_with_smoothing: int = 331,
                 seed=1903):
        """
        Initialize the model for optimizing the unknown refractive index `n`.

        Args:
            incoming_medium: Material at the entrance of the stack.
            outgoing_medium: Material at the exit of the stack.
            material_dist_above_unk_in_str: List of material strings above the unknown layer.
            material_dist_below_unk_in_str: List of material strings below the unknown layer.
            thickness_above_unk: Thickness list for layers above the unknown material.
            thickness_below_unk: Thickness list for layers below the unknown material.
            light: Light object containing wavelength, angle, polarization data.
            coherency_list: Coherency settings for each interface.
            min_refractive_index: Lower bound of refractive index search.
            max_refractive_index: Upper bound of refractive index search.
            observable: The target metric (e.g., reflectance, transmittance) for optimization.
            smoothing_method: Smoothing method applied to refractive index ('gaussian', etc.).
            smoothing_window_range: Range of the smoothing kernel window.
            smoothing_sigma: Sigma value for Gaussian smoothing.
            smoothing_amplitude: Amplitude of smoothing kernel.
            smoothing_kernel_size: Size of smoothing kernel window.
            smoothing_decay: Decay factor for exponential smoothing (if used).
            num_of_data_points: Number of points for refractive index profile.
            num_of_data_points_with_smoothing: Points used before padding.
            seed: Random seed for reproducibility.
        """

        # Construct the complete list of mediums across the stack
        num_of_mediums_above_unk = len([incoming_medium] + material_dist_above_unk_in_str)
        mediums = [incoming_medium] + material_dist_above_unk_in_str + material_dist_below_unk_in_str + [outgoing_medium]

        # Convert material names into distribution index and database set
        fixed_material_set, fixed_material_distribution = material_distribution_to_set(mediums)

        self.num_of_data_points = num_of_data_points
        self.fixed_data = create_data(fixed_material_set)  # Database of known materials' nk data
        self.thickness_above_unk = thickness_above_unk
        self.thickness_below_unk = thickness_below_unk
        self.key = PRNGKey(seed)

        # Light configuration
        self.wavelengths = light.wavelength
        self.angle_of_incidences = light.angle_of_incidence
        self.coherency_list = coherency_list
        self.polarization = light.polarization

        self.max_data_dim = self.fixed_data.shape[2]
        self.num_of_repeat = self.max_data_dim - num_of_data_points_with_smoothing

        # Randomly initialize unknown refractive index array within bounds
        self.refractive_index = refractive_index_initilizer(
            self.key, min_refractive_index, max_refractive_index, num_of_data_points
        )

        # Wavelengths for the unknown layer (interpolated/smoothed)
        self.dynamic_layer_wavelengths = jnp.linspace(
            jnp.min(self.wavelengths), jnp.max(self.wavelengths), num_of_data_points_with_smoothing
        )
        self.dynamic_layer_wavelengths = repeat_last_element(self.dynamic_layer_wavelengths, self.num_of_repeat)

        # Observable function is mapped to (R,T,A) → observable (e.g., RMSE)
        self.observable_func = objective_to_function(observable)

        # Insert placeholder index for unknown layer in material distribution
        self.material_distribution = jnp.insert(
            fixed_material_distribution, num_of_mediums_above_unk, len(fixed_material_distribution)
        )

        # Bounds and smoothing configuration
        self.min_refractive_index = min_refractive_index
        self.max_refractive_index = max_refractive_index
        self.smoothing_method = smoothing_method
        self.smoothing_window_range = smoothing_window_range
        self.smoothing_sigma = smoothing_sigma
        self.smoothing_amplitude = smoothing_amplitude
        self.smoothing_kernel_size = smoothing_kernel_size
        self.smoothing_decay = smoothing_decay

    def __call__(self, unknown_layer_thickness):
        """
        Forward pass of the Equinox model. Returns the observable 
        quantity calculated from the optical simulation.

        Args:
            unknown_layer_thickness: Thickness of the layer with unknown refractive index.

        Returns:
            Observable result (float or array depending on observable) calculated 
            using transfer matrix method (TMM).
        """

        # Apply smoothing to the learned refractive index to ensure continuity
        smoothed_refractive_index = smooth_nk(
            nk=self.refractive_index,
            method=self.smoothing_method,
            window_range=self.smoothing_window_range,
            sigma=self.smoothing_sigma,
            amplitude=self.smoothing_amplitude,
            kernel_size=self.smoothing_kernel_size,
            decay=self.smoothing_decay
        )

        # Pad refractive index array to match maximum data dimension
        refractive_index = repeat_last_element(smoothed_refractive_index, self.num_of_repeat)

        # Merge the known nk data and the unknown refractive index into one array
        data = merge_n_data(self.fixed_data, refractive_index, self.dynamic_layer_wavelengths, self.max_data_dim)

        # Merge thickness values for full layer stack
        thickness_list = merge_thickness(unknown_layer_thickness, self.thickness_above_unk, self.thickness_below_unk)

        # Simulate optical response using the Transfer Matrix Method (TMM)
        R, T = tmm_insensitive(
            data,
            self.material_distribution,
            thickness_list,
            self.wavelengths,
            self.angle_of_incidences,
            self.coherency_list,
            self.polarization
        )

        # Compute absorption A = 1 - R - T
        A = jnp.subtract(1, jnp.add(R, T))

        # Compute final observable from (R, T, A)
        observable_result = self.observable_func(R, T, A)

        return observable_result


class OneUnkMaterialStack_NK(eqx.Module):
    """
    Equinox model for extracting both the refractive index (n) and extinction coefficient (k)
    of a single unknown material layer in a multilayer thin film stack using experimental data.
    This class is designed for gradient-based optimization and uses smoothing for regularization.

    Attributes:
        refractive_index (Array): Trainable parameter representing the refractive index (n) of the unknown layer.
        extinction_coefficient (Array): Trainable parameter representing the extinction coefficient (k).
        num_of_data_points (int): Number of data points to be optimized for the unknown layer.
        num_of_repeat (int): Padding size to match data length with the max dimension required by the stack.
        max_data_dim (int): The maximum number of wavelength points used in the TMM stack.
        material_distribution (Array): Encoded material sequence for the entire stack including unknown material.
        fixed_data (Array): Optical data for the fixed (known) layers in the stack.
        thickness_above_unk (Array): Thickness values of the layers above the unknown material.
        thickness_below_unk (Array): Thickness values of the layers below the unknown material.
        wavelengths (Array): Experimental wavelengths.
        dynamic_layer_wavelengths (Array): Interpolated wavelength grid used for smoothing the unknown layer.
        angle_of_incidences (Array): Angles at which light hits the thin film stack.
        coherency_list (Array): Describes whether each interface is coherent or incoherent.
        polarization (Array): Describes the polarization state of the incident light (s, p, or mixed).
        key (Array): PRNG key for parameter initialization.
        observable_func (Callable): Function that computes observable (e.g., reflectance) from R, T, A.
        min_refractive_index (float): Lower bound for refractive index.
        max_refractive_index (float): Upper bound for refractive index.
        min_extinction_coeff (float): Lower bound for extinction coefficient.
        max_extinction_coeff (float): Upper bound for extinction coefficient.
        smoothing_method (str): Smoothing technique name (e.g., "gaussian").
        smoothing_window_range (Tuple[float, float]): Range for the smoothing kernel.
        smoothing_sigma (float): Standard deviation used in smoothing kernel.
        smoothing_amplitude (float): Amplitude for controlling smoothing intensity.
        smoothing_kernel_size (int): Kernel size for smoothing.
        smoothing_decay (float): Decay factor for the smoothing kernel.
    """

    # All fields are defined with Equinox-style static and dynamic parameters
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
    smoothing_method: str = eqx.static_field()
    smoothing_window_range: Tuple[float, float] = eqx.static_field()
    smoothing_sigma: float = eqx.static_field()
    smoothing_amplitude: float = eqx.static_field()
    smoothing_kernel_size: int = eqx.static_field()
    smoothing_decay: float = eqx.static_field()

    def __init__(self,
                 incoming_medium,
                 outgoing_medium,
                 material_dist_above_unk_in_str,
                 material_dist_below_unk_in_str,
                 thickness_above_unk,
                 thickness_below_unk,
                 light,
                 coherency_list,
                 min_refractive_index,
                 max_refractive_index,
                 max_extinction_coeff,
                 min_extinction_coeff,
                 observable,
                 smoothing_method: str = "gaussian",
                 smoothing_window_range: Tuple[float, float] = (-10.0, 10.0),
                 smoothing_sigma: float = 4.0,
                 smoothing_amplitude: float = 0.5,
                 smoothing_kernel_size: int = 20,
                 smoothing_decay: float = 0.8,
                 num_of_data_points: int = 350,
                 num_of_data_points_with_smoothing: int = 331,
                 seed=1903):
        """
        Initialize the OneUnkMaterialStack_NK model.

        Args:
            incoming_medium: Material name or identifier of the medium before the first layer.
            outgoing_medium: Material name or identifier of the medium after the last layer.
            material_dist_above_unk_in_str: List of material names above the unknown layer.
            material_dist_below_unk_in_str: List of material names below the unknown layer.
            thickness_above_unk: Thickness array of layers above the unknown material.
            thickness_below_unk: Thickness array of layers below the unknown material.
            light: Light source object with wavelength, angle_of_incidence, and polarization.
            coherency_list: List defining coherency for each interface in the stack.
            min_refractive_index: Lower bound of the unknown layer's refractive index.
            max_refractive_index: Upper bound of the unknown layer's refractive index.
            max_extinction_coeff: Upper bound of the extinction coefficient.
            min_extinction_coeff: Lower bound of the extinction coefficient.
            observable: Function name that computes the desired observable (e.g., "R", "T", or "A").
            smoothing_method: Smoothing technique (default: "gaussian").
            smoothing_window_range: Range of the smoothing filter window.
            smoothing_sigma: Sigma for Gaussian smoothing.
            smoothing_amplitude: Amplitude of smoothing.
            smoothing_kernel_size: Size of the kernel used for smoothing.
            smoothing_decay: Decay rate in the smoothing kernel.
            num_of_data_points: Number of points used for optimization.
            num_of_data_points_with_smoothing: Number of wavelength points after smoothing.
            seed: Random seed for reproducibility.
        """

        # Compute total number of layers above the unknown one (including input medium)
        num_of_mediums_above_unk = len([incoming_medium] + material_dist_above_unk_in_str)

        # Combine all materials into a single sequence: input → above → below → output
        mediums = [incoming_medium] + material_dist_above_unk_in_str + material_dist_below_unk_in_str + [outgoing_medium]

        # Convert material names into numerical identifiers and get fixed structure
        fixed_material_set, fixed_material_distribution = material_distribution_to_set(mediums)

        self.num_of_data_points = num_of_data_points
        self.fixed_data = create_data(fixed_material_set)

        # Save thicknesses for all known layers
        self.thickness_above_unk = thickness_above_unk
        self.thickness_below_unk = thickness_below_unk

        # Seeded key for random number generation
        self.key = PRNGKey(seed)

        # Light properties
        self.wavelengths = light.wavelength
        self.angle_of_incidences = light.angle_of_incidence
        self.coherency_list = coherency_list
        self.polarization = light.polarization

        # Get dimensions for padding/repetition
        self.max_data_dim = self.fixed_data.shape[2]
        self.num_of_repeat = self.max_data_dim - num_of_data_points_with_smoothing

        # Initialize trainable unknown parameters
        self.refractive_index = refractive_index_initilizer(self.key, min_refractive_index, max_refractive_index, num_of_data_points)
        self.extinction_coefficient = extinction_coefficient_initilizer(self.key, min_extinction_coeff, max_extinction_coeff, num_of_data_points)

        # Interpolated wavelength range for smoothing
        self.dynamic_layer_wavelengths = jnp.linspace(jnp.min(self.wavelengths), jnp.max(self.wavelengths), num_of_data_points_with_smoothing)
        self.dynamic_layer_wavelengths = repeat_last_element(self.dynamic_layer_wavelengths, self.num_of_repeat)

        # Convert observable string to actual function
        self.observable_func = objective_to_function(observable)

        # Construct the full material distribution and insert the unknown layer
        self.material_distribution = jnp.insert(fixed_material_distribution, num_of_mediums_above_unk, len(fixed_material_distribution))

        # Store bounds and smoothing parameters
        self.min_refractive_index = min_refractive_index
        self.max_refractive_index = max_refractive_index
        self.min_extinction_coeff = min_extinction_coeff
        self.max_extinction_coeff = max_extinction_coeff
        self.smoothing_method = smoothing_method
        self.smoothing_window_range = smoothing_window_range
        self.smoothing_sigma = smoothing_sigma
        self.smoothing_amplitude = smoothing_amplitude
        self.smoothing_kernel_size = smoothing_kernel_size
        self.smoothing_decay = smoothing_decay

    def __call__(self, unknown_layer_thickness):
        """
        Forward pass of the model to compute observable based on current parameters.

        Args:
            unknown_layer_thickness (Array): The thickness of the unknown layer to be optimized.

        Returns:
            observable_result (Array): Computed observable (e.g., reflectance, transmittance, absorbance).
        """

        # Smooth both refractive index and extinction coefficient
        smoothed_refractive_index = smooth_nk(
            nk=self.refractive_index,
            method=self.smoothing_method,
            window_range=self.smoothing_window_range,
            sigma=self.smoothing_sigma,
            amplitude=self.smoothing_amplitude,
            kernel_size=self.smoothing_kernel_size,
            decay=self.smoothing_decay
        )

        smoothed_extinction_coefficient = smooth_nk(
            nk=self.extinction_coefficient,
            method=self.smoothing_method,
            window_range=self.smoothing_window_range,
            sigma=self.smoothing_sigma,
            amplitude=self.smoothing_amplitude,
            kernel_size=self.smoothing_kernel_size,
            decay=self.smoothing_decay
        )

        # Pad smoothed values to match the TMM data shape
        refractive_index = repeat_last_element(smoothed_refractive_index, self.num_of_repeat)
        extinction_coefficient = repeat_last_element(smoothed_extinction_coefficient, self.num_of_repeat)

        # Merge known and unknown material optical data
        data = merge_nk_data(
            self.fixed_data,
            refractive_index,
            extinction_coefficient,
            self.dynamic_layer_wavelengths,
            self.max_data_dim
        )

        # Combine all layer thicknesses
        thickness_list = merge_thickness(
            unknown_layer_thickness,
            self.thickness_above_unk,
            self.thickness_below_unk
        )

        # Simulate R, T using the Transfer Matrix Method
        R, T = tmm_insensitive(
            data,
            self.material_distribution,
            thickness_list,
            self.wavelengths,
            self.angle_of_incidences,
            self.coherency_list,
            self.polarization
        )

        # Compute absorbance A = 1 - R - T
        A = jnp.subtract(1, jnp.add(R, T))

        # Compute and return final observable using user-defined function
        observable_result = self.observable_func(R, T, A)

        return observable_result


class OneUnkMaterialSparseStack_N(eqx.Module):
    """
    A differentiable Equinox model for modeling and optimizing the **refractive index** 
    of a *single unknown sparse thin film layer* (e.g., WO_(3-x)) in a multilayer optical stack, 
    using coherent Transfer Matrix Method (TMM) and smoothing techniques.

    This model enables gradient-based optimization of the unknown material’s refractive index profile
    by matching predicted optical responses (R, T, A) to experimental data.

    Attributes:
        refractive_index (Array): Trainable refractive index array for the unknown layer.
        compression_ratio (Array): A scalar array to scale the unknown layer’s thickness.
        num_of_data_points (int): Number of wavelength samples for the unknown layer.
        num_of_repeat (int): Number of points to pad the wavelength dimension (for smoothing alignment).
        max_data_dim (int): Maximum wavelength sampling resolution across all layers.
        material_distribution (Array): Encoded index-based material order for TMM simulation.
        fixed_data (Array): Precomputed data (n, k) for all known materials in the stack.
        thickness_above_unk (Array): Thicknesses of layers above the unknown material.
        thickness_below_unk (Array): Thicknesses of layers below the unknown material.
        wavelengths (Array): Experimental wavelength values.
        dynamic_layer_wavelengths (Array): Interpolated wavelength values for the unknown layer.
        angle_of_incidences (Array): Incidence angles (degrees or radians).
        polarization (Array): Polarization mode(s) (e.g., "s", "p", "unpolarized").
        key (Array): PRNG key used for initialization.
        observable_func (Callable): Objective function to extract desired optical quantities from R, T, A.
        min_refractive_index (Array): Lower bound for unknown layer refractive index.
        max_refractive_index (Array): Upper bound for unknown layer refractive index.
        min_compression_ratio (float): Lower bound for scaling the unknown layer's thickness.
        max_compression_ratio (float): Upper bound for scaling the unknown layer's thickness.
        smoothing_method (str): Method used for smoothing (e.g., "gaussian").
        smoothing_window_range (Tuple[float, float]): Range of the smoothing window.
        smoothing_sigma (float): Sigma parameter for Gaussian smoothing.
        smoothing_amplitude (float): Amplitude of smoothing kernel.
        smoothing_kernel_size (int): Size of the smoothing kernel.
        smoothing_decay (float): Exponential decay for kernel weighting.
    """

    # Attributes (declared as static or dynamic fields in Equinox)
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
    min_compression_ratio: float = eqx.static_field()
    max_compression_ratio: float = eqx.static_field()
    smoothing_method: str = eqx.static_field()
    smoothing_window_range: Tuple[float, float] = eqx.static_field()
    smoothing_sigma: float = eqx.static_field()
    smoothing_amplitude: float = eqx.static_field()
    smoothing_kernel_size: int = eqx.static_field()
    smoothing_decay: float = eqx.static_field()

    def __init__(self,
                 incoming_medium,
                 outgoing_medium,
                 material_dist_above_unk_in_str,
                 material_dist_below_unk_in_str,
                 thickness_above_unk,
                 thickness_below_unk,
                 light,
                 min_refractive_index,
                 max_refractive_index,
                 min_compression_ratio,
                 max_compression_ratio,
                 observable,
                 smoothing_method: str = "gaussian",
                 smoothing_window_range: Tuple[float, float] = (-10.0, 10.0),
                 smoothing_sigma: float = 4.0,
                 smoothing_amplitude: float = 0.5,
                 smoothing_kernel_size: int = 20,
                 smoothing_decay: float = 0.8,
                 num_of_data_points: int = 350,
                 num_of_data_points_with_smoothing: int = 331,
                 seed=1903):
        """
        Initialize the OneUnkMaterialSparseStack_N model.

        Args:
            incoming_medium: Material name or identifier for the incident medium.
            outgoing_medium: Material name or identifier for the transmission medium.
            material_dist_above_unk_in_str: List of materials above the unknown layer.
            material_dist_below_unk_in_str: List of materials below the unknown layer.
            thickness_above_unk: Thicknesses of the above layers (µm).
            thickness_below_unk: Thicknesses of the below layers (µm).
            light: An object holding wavelengths, AOIs, and polarization.
            min_refractive_index: Minimum initial guess for unknown layer's n.
            max_refractive_index: Maximum initial guess for unknown layer's n.
            min_compression_ratio: Lower limit of compression ratio scaling.
            max_compression_ratio: Upper limit of compression ratio scaling.
            observable: Objective function identifier to be turned into callable.
            smoothing_method: Method for refractive index smoothing.
            smoothing_window_range: Range for smoothing filter window.
            smoothing_sigma: Sigma value for Gaussian smoothing kernel.
            smoothing_amplitude: Amplitude for smoothing function.
            smoothing_kernel_size: Number of points in smoothing kernel.
            smoothing_decay: Decay rate for smoothing weight function.
            num_of_data_points: Number of wavelength points for unknown layer.
            num_of_data_points_with_smoothing: Number of points before padding.
            seed: Random seed for reproducibility.
        """
        num_of_mediums_above_unk = len([incoming_medium] + material_dist_above_unk_in_str)
        # Combine all mediums into a full stack for index encoding
        mediums = [incoming_medium] + material_dist_above_unk_in_str + material_dist_below_unk_in_str + [outgoing_medium]
        fixed_material_set, fixed_material_distribution = material_distribution_to_set(mediums)

        self.num_of_data_points = num_of_data_points
        self.fixed_data = create_data(fixed_material_set)
        self.thickness_above_unk = thickness_above_unk
        self.thickness_below_unk = thickness_below_unk

        # PRNG key for initialization
        self.key = PRNGKey(seed)

        # Extract experimental measurement data
        self.wavelengths = light.wavelength
        self.angle_of_incidences = light.angle_of_incidence
        self.polarization = light.polarization

        self.max_data_dim = self.fixed_data.shape[2]
        self.num_of_repeat = self.max_data_dim - num_of_data_points_with_smoothing

        # Initialize refractive index of unknown layer
        self.refractive_index = refractive_index_initilizer(self.key, min_refractive_index, max_refractive_index, num_of_data_points)

        # Prepare wavelength domain for unknown layer and pad it
        self.dynamic_layer_wavelengths = jnp.linspace(jnp.min(self.wavelengths), jnp.max(self.wavelengths), num_of_data_points_with_smoothing)
        self.dynamic_layer_wavelengths = repeat_last_element(self.dynamic_layer_wavelengths, self.num_of_repeat)

        self.compression_ratio = jnp.array([1.0])

        self.observable_func = objective_to_function(observable)

        # Insert unknown material at correct index in material distribution
        self.material_distribution = jnp.insert(fixed_material_distribution, num_of_mediums_above_unk, len(fixed_material_distribution))

        # Assign bounds
        self.min_refractive_index = min_refractive_index
        self.max_refractive_index = max_refractive_index
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio

        # Store smoothing parameters
        self.smoothing_method = smoothing_method
        self.smoothing_window_range = smoothing_window_range
        self.smoothing_sigma = smoothing_sigma
        self.smoothing_amplitude = smoothing_amplitude
        self.smoothing_kernel_size = smoothing_kernel_size
        self.smoothing_decay = smoothing_decay

    def __call__(self, unknown_layer_thickness):
        """
        Forward pass through the model. Computes the optical response and applies the objective.

        Args:
            unknown_layer_thickness (Array): Thickness of unknown layer (in nanometers).

        Returns:
            float: Output of observable function (e.g., loss value or physical observable).
        """
        # Scale the unknown thickness by the compression ratio (convert from nm to µm)
        unknown_layer_thickness = jnp.multiply(jnp.true_divide(unknown_layer_thickness, 1000), self.compression_ratio)

        # Apply smoothing to the refractive index of the unknown layer
        smoothed_refractive_index = smooth_nk(
            nk=self.refractive_index,
            method=self.smoothing_method,
            window_range=self.smoothing_window_range,
            sigma=self.smoothing_sigma,
            amplitude=self.smoothing_amplitude,
            kernel_size=self.smoothing_kernel_size,
            decay=self.smoothing_decay,
        )

        # Pad the smoothed n values to match the fixed data dimensions
        refractive_index = repeat_last_element(smoothed_refractive_index, self.num_of_repeat)

        # Merge fixed material data with unknown layer n data
        data = merge_n_data(self.fixed_data, refractive_index, self.dynamic_layer_wavelengths, self.max_data_dim)

        # Construct thickness list including unknown, above, and below layers
        thickness_list = merge_thickness(unknown_layer_thickness, self.thickness_above_unk, self.thickness_below_unk)

        # Apply coherent TMM for sparse thin films
        result = coh_tmm_sparse(data, self.material_distribution, thickness_list,
                                self.wavelengths, self.angle_of_incidences, self.polarization)

        # Compute objective function (e.g., match to experiment)
        observable_result = self.observable_func(
            R=result.at[0].get(), 
            T=result.at[1].get(), 
            A=result.at[2].get()
        )

        return observable_result


class OneUnkMaterialSparseStack_NK(eqx.Module):
    """
    Equinox model for optimizing the refractive index (n) and extinction coefficient (k)
    of a **sparse (like WO_{3-x}) unknown material** layer in a multilayer optical thin-film stack.
    It simulates the optical response and enables gradient-based learning with experimental data.

    Attributes:
        refractive_index (Array): The trainable refractive index `n(λ)` array of the unknown material.
        extinction_coefficient (Array): The trainable extinction coefficient `k(λ)` array of the unknown material.
        compression_ratio (Array): Scaling factor applied to unknown layer thickness during optimization.
        num_of_data_points (int): Number of wavelength samples used for refractive index and extinction coefficient.
        num_of_repeat (int): Number of times the smoothed n/k arrays are repeated to match simulation dimension.
        max_data_dim (int): Dimensionality of the simulation data across all materials.
        material_distribution (Array): Encoded integer-based stack structure defining layer arrangement.
        fixed_data (Array): Optical data (n, k) of known materials in the multilayer stack.
        thickness_above_unk (Array): Thicknesses of layers above the unknown material.
        thickness_below_unk (Array): Thicknesses of layers below the unknown material.
        wavelengths (Array): Target wavelengths for simulation.
        dynamic_layer_wavelengths (Array): Wavelengths for smoothed/learned unknown material.
        angle_of_incidences (Array): Angle(s) of light incidence for the simulation.
        polarization (Array): Polarization type(s) of incident light.
        key (Array): JAX PRNG key for initialization.
        observable_func (Callable): Objective function that maps (R, T, A) to scalar loss.
        min_refractive_index (float): Lower bound for refractive index during init/optimization.
        max_refractive_index (float): Upper bound for refractive index during init/optimization.
        min_extinction_coeff (float): Lower bound for extinction coefficient during init/optimization.
        max_extinction_coeff (float): Upper bound for extinction coefficient during init/optimization.
        min_compression_ratio (float): Lower bound for compression ratio.
        max_compression_ratio (float): Upper bound for compression ratio.
        smoothing_method (str): Smoothing method for the unknown material (e.g., 'gaussian').
        smoothing_window_range (Tuple[float, float]): Range for smoothing window.
        smoothing_sigma (float): Sigma parameter for Gaussian smoothing.
        smoothing_amplitude (float): Amplitude factor for smoothing.
        smoothing_kernel_size (int): Kernel size used in smoothing function.
        smoothing_decay (float): Decay factor used in smoothing.
    """

    # === Learnable Fields ===
    refractive_index: Array
    extinction_coefficient: Array
    compression_ratio: Array

    # === Static Fields (Fixed) ===
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
    min_compression_ratio: float = eqx.static_field()
    max_compression_ratio: float = eqx.static_field()
    smoothing_method: str = eqx.static_field()
    smoothing_window_range: Tuple[float, float] = eqx.static_field()
    smoothing_sigma: float = eqx.static_field()
    smoothing_amplitude: float = eqx.static_field()
    smoothing_kernel_size: int = eqx.static_field()
    smoothing_decay: float = eqx.static_field()

    def __init__(
        self, 
        incoming_medium,
        outgoing_medium,
        material_dist_above_unk_in_str,
        material_dist_below_unk_in_str,
        thickness_above_unk,
        thickness_below_unk,
        light,
        min_refractive_index,
        max_refractive_index,
        max_extinction_coeff,
        min_extinction_coeff,
        min_compression_ratio,
        max_compression_ratio,
        observable,
        smoothing_method: str = "gaussian",
        smoothing_window_range: Tuple[float, float] = (-10.0, 10.0),
        smoothing_sigma: float = 4.0,
        smoothing_amplitude: float = 0.5,
        smoothing_kernel_size: int = 20,
        smoothing_decay: float = 0.8,
        num_of_data_points: int = 350,
        num_of_data_points_with_smoothing: int = 331,
        seed=1903
    ):
        """
        Initializes the model with the known layer properties and unknown layer bounds.

        Args:
            incoming_medium: Material string or identifier for the entrance medium (e.g., 'air').
            outgoing_medium: Material string or identifier for the exit medium (e.g., 'glass').
            material_dist_above_unk_in_str: List of material strings above the unknown material.
            material_dist_below_unk_in_str: List of material strings below the unknown material.
            thickness_above_unk: Thicknesses of the above layers (in nm).
            thickness_below_unk: Thicknesses of the below layers (in nm).
            light: A configuration object containing wavelength, angle of incidence, and polarization.
            min_refractive_index: Lower bound for n during initialization.
            max_refractive_index: Upper bound for n during initialization.
            max_extinction_coeff: Upper bound for k during initialization.
            min_extinction_coeff: Lower bound for k during initialization.
            min_compression_ratio: Minimum scaling ratio for unknown thickness.
            max_compression_ratio: Maximum scaling ratio for unknown thickness.
            observable: String or callable representing the loss objective (e.g., 'mean_squared_error').
            smoothing_method: Method to smooth n/k values ('gaussian', 'savgol', etc.).
            smoothing_window_range: Tuple defining smoothing range.
            smoothing_sigma: Sigma for smoothing kernel.
            smoothing_amplitude: Amplitude of smoothing.
            smoothing_kernel_size: Kernel size for smoothing.
            smoothing_decay: Exponential decay rate in smoothing.
            num_of_data_points: Number of target wavelengths for fitting n and k.
            num_of_data_points_with_smoothing: Reduced number of smoothed points for learning.
            seed: Random seed for reproducibility.
        """
        # Construct the layer ordering: input → above → unknown → below → output
        num_of_mediums_above_unk = len([incoming_medium] + material_dist_above_unk_in_str)
        mediums = [incoming_medium] + material_dist_above_unk_in_str + material_dist_below_unk_in_str + [outgoing_medium]

        # Convert material names to unique integer set and distribution encoding
        fixed_material_set, fixed_material_distribution = material_distribution_to_set(mediums)
        
        self.num_of_data_points = num_of_data_points
        self.fixed_data = create_data(fixed_material_set)
        self.thickness_above_unk = thickness_above_unk
        self.thickness_below_unk = thickness_below_unk

        # RNG key for parameter initialization
        self.key = PRNGKey(seed)

        # Load optical simulation conditions
        self.wavelengths = light.wavelength
        self.angle_of_incidences = light.angle_of_incidence
        self.polarization = light.polarization

        # Determine data shape parameters
        self.max_data_dim = self.fixed_data.shape[2]
        self.num_of_repeat = self.max_data_dim - num_of_data_points_with_smoothing

        # Initialize learnable n and k values
        self.refractive_index = refractive_index_initilizer(self.key, min_refractive_index, max_refractive_index, num_of_data_points)
        self.extinction_coefficient = extinction_coefficient_initilizer(self.key, min_extinction_coeff, max_extinction_coeff, num_of_data_points)

        # Define wavelength grid for unknown material
        self.dynamic_layer_wavelengths = jnp.linspace(jnp.min(self.wavelengths), jnp.max(self.wavelengths), num_of_data_points_with_smoothing)
        self.dynamic_layer_wavelengths = repeat_last_element(self.dynamic_layer_wavelengths, self.num_of_repeat)

        # Set initial compression ratio for layer thickness scaling
        self.compression_ratio = jnp.array([1.0])

        # Convert observable to callable function
        self.observable_func = objective_to_function(observable)

        # Insert the unknown material (special index) into the encoded distribution
        self.material_distribution = jnp.insert(fixed_material_distribution, num_of_mediums_above_unk, len(fixed_material_distribution))

        # Store bounds and smoothing parameters
        self.min_refractive_index = min_refractive_index
        self.max_refractive_index = max_refractive_index
        self.min_extinction_coeff = min_extinction_coeff
        self.max_extinction_coeff = max_extinction_coeff
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio
        self.smoothing_method = smoothing_method
        self.smoothing_window_range = smoothing_window_range
        self.smoothing_sigma = smoothing_sigma
        self.smoothing_amplitude = smoothing_amplitude
        self.smoothing_kernel_size = smoothing_kernel_size
        self.smoothing_decay = smoothing_decay

    def __call__(self, unknown_layer_thickness: Array) -> float:
        """
        Evaluates the observable (e.g., loss or measurement) for the current stack configuration.

        Args:
            unknown_layer_thickness (Array): Thickness of the unknown material layer (in nm).

        Returns:
            float: The value of the observable (e.g., MSE, reflection ratio) computed from the TMM.
        """
        # Scale the unknown thickness by the compression factor
        unknown_layer_thickness = jnp.multiply(jnp.true_divide(unknown_layer_thickness, 1000), self.compression_ratio)

        # Apply smoothing to learned n(λ)
        smoothed_refractive_index = smooth_nk(
            nk=self.refractive_index,
            method=self.smoothing_method,
            window_range=self.smoothing_window_range,
            sigma=self.smoothing_sigma,
            amplitude=self.smoothing_amplitude,
            kernel_size=self.smoothing_kernel_size,
            decay=self.smoothing_decay
        )

        # Apply smoothing to learned k(λ)
        smoothed_extinction_coefficient = smooth_nk(
            nk=self.extinction_coefficient,
            method=self.smoothing_method,
            window_range=self.smoothing_window_range,
            sigma=self.smoothing_sigma,
            amplitude=self.smoothing_amplitude,
            kernel_size=self.smoothing_kernel_size,
            decay=self.smoothing_decay
        )

        # Extend n and k to match full simulation dimension
        refractive_index = repeat_last_element(smoothed_refractive_index, self.num_of_repeat)
        extinction_coefficient = repeat_last_element(smoothed_extinction_coefficient, self.num_of_repeat)

        # Merge learned material with fixed data
        data = merge_nk_data(self.fixed_data, refractive_index, extinction_coefficient, self.dynamic_layer_wavelengths, self.max_data_dim)

        # Merge thickness vectors into full stack configuration
        thickness_list = merge_thickness(unknown_layer_thickness, self.thickness_above_unk, self.thickness_below_unk)

        # Compute reflectance (R), transmittance (T), and absorbance (A) using sparse TMM
        result = coh_tmm_sparse(data, self.material_distribution, thickness_list, self.wavelengths, self.angle_of_incidences, self.polarization)

        # Evaluate and return the observable (loss)
        observable_result = self.observable_func(R=result.at[0].get(), T=result.at[1].get(), A=result.at[2].get())

        return observable_result