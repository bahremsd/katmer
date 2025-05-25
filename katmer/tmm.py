import jax.numpy as jnp  # Import JAX's NumPy wrapper for GPU/TPU-compatible numerical operations
from jax.lax import fori_loop  # JAX's functional for-loop replacement for efficient iteration
from jax import Array  # Used for type annotation, represents JAX arrays
from jax.typing import ArrayLike  # Used for type annotations to accept JAX-compatible array inputs
from typing import Callable, Tuple  # Python's standard typing module

# tmmax and local imports for thin-film modeling utilities
from tmmax.tmm import vectorized_coh_tmm, tmm
from .initializers import generate_deposition_samples


def tmm_s_or_p_pol_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization):
    """
    Computes reflection (R), transmission (T), and absorption (A) using the TMM for a given
    single polarization (s or p).

    Args:
        data: Optical data needed for simulation.
        material_distribution: Sequence of materials forming the stack.
        thickness_list: List of layer thicknesses.
        wavelengths: Wavelengths to evaluate.
        angle_of_incidences: Incident angles for the simulation.
        coherency_list: Coherency info for each interface.
        polarization: Boolean array indicating polarization (False = s, True = p).

    Returns:
        A stacked JAX array [R, T, A] for given polarization.
    """
    R, T = tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization)
    A = jnp.subtract(1, jnp.add(R, T))  # A = 1 - R - T
    return jnp.array([R, T, A])


def tmm_u_pol_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list):
    """
    Computes polarization-insensitive (unpolarized) R, T, A values by averaging s- and p-polarization results.

    Args:
        Same as above, excluding polarization.

    Returns:
        A stacked JAX array [R, T, A] averaged over s and p polarizations.
    """
    R_s, T_s = tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization=jnp.array([False], dtype=bool))
    R_p, T_p = tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization=jnp.array([True], dtype=bool))
    R = jnp.true_divide(R_s + R_p, 2)
    T = jnp.true_divide(T_s + T_p, 2)
    A = jnp.subtract(1, jnp.add(R, T))
    return jnp.array([R, T, A])


def tmm_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization):
    """
    Wrapper that chooses s, p, or unpolarized insensitive TMM simulation based on polarization flag.

    Args:
        polarization: Integer-coded array — 0 for s-pol, 1 for p-pol, 2 for unpolarized.

    Returns:
        A stacked JAX array [R, T, A].
    """
    result = jnp.select(
        condlist=[
            jnp.array_equal(polarization, jnp.array([0], dtype=jnp.int32)),
            jnp.array_equal(polarization, jnp.array([1], dtype=jnp.int32)),
            jnp.array_equal(polarization, jnp.array([2], dtype=jnp.int32))
        ],
        choicelist=[
            tmm_s_or_p_pol_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, jnp.array([False], dtype=bool)),
            tmm_s_or_p_pol_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, jnp.array([True], dtype=bool)),
            tmm_u_pol_insensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list)
        ]
    )
    return result


def tmm_s_or_p_pol_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization, wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num):
    """
    Sensitivity-aware TMM for single-polarization, accounting for fabrication uncertainties by sampling perturbed layer thicknesses.

    Returns:
        Mean R, T, A over sensitivity_optimization_sample_num samples.
    """
    num_wavelengths, num_angles = wl_angle_shape
    R_T_results = jnp.empty((sensitivity_optimization_sample_num, num_angles, num_wavelengths, 2))

    deposition_samples = generate_deposition_samples(key, thickness_list, deposition_deviation_percentage, sensitivity_optimization_sample_num)

    def one_deviation_tmm(i, R_T_results):
        R, T = tmm(data, material_distribution, deposition_samples.at[i, :].get(), wavelengths, angle_of_incidences, coherency_list, polarization)
        R_T_results = R_T_results.at[i, :, :, 0].set(R)
        R_T_results = R_T_results.at[i, :, :, 1].set(T)
        return R_T_results

    R_T_results = fori_loop(0, sensitivity_optimization_sample_num, one_deviation_tmm, R_T_results)
    R_T_mean = jnp.mean(R_T_results, axis=0)
    R, T = R_T_mean[:, :, 0], R_T_mean[:, :, 1]
    A = jnp.subtract(1, jnp.add(R, T))
    return jnp.array([R, T, A])


def tmm_u_pol_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num):
    """
    Sensitivity-aware TMM for unpolarized light, averaging over s and p polarizations.

    Returns:
        Mean R, T, A values across thickness deviations and both polarizations.
    """
    num_wavelengths, num_angles = wl_angle_shape
    R_T_results = jnp.empty((sensitivity_optimization_sample_num, num_angles, num_wavelengths, 2))
    deposition_samples = generate_deposition_samples(key, thickness_list, deposition_deviation_percentage, sensitivity_optimization_sample_num)

    def one_deviation_tmm(i, R_T_results):
        R_s, T_s = tmm(data, material_distribution, deposition_samples.at[i, :].get(), wavelengths, angle_of_incidences, coherency_list, jnp.array([False], dtype=bool))
        R_p, T_p = tmm(data, material_distribution, deposition_samples.at[i, :].get(), wavelengths, angle_of_incidences, coherency_list, jnp.array([True], dtype=bool))
        R = jnp.true_divide(R_s + R_p, 2)
        T = jnp.true_divide(T_s + T_p, 2)
        R_T_results = R_T_results.at[i, :, :, 0].set(R)
        R_T_results = R_T_results.at[i, :, :, 1].set(T)
        return R_T_results

    R_T_results = fori_loop(0, sensitivity_optimization_sample_num, one_deviation_tmm, R_T_results)
    R_T_mean = jnp.mean(R_T_results, axis=0)
    R, T = R_T_mean[:, :, 0], R_T_mean[:, :, 1]
    A = jnp.subtract(1, jnp.add(R, T))
    return jnp.array([R, T, A])


def tmm_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, polarization, wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num):
    """
    Dispatcher function for TMM simulation under sensitivity analysis, based on polarization mode.

    Returns:
        Averaged [R, T, A] for given polarization and perturbed thickness samples.
    """
    return jnp.select(
        condlist=[
            jnp.array_equal(polarization, jnp.array([0], dtype=jnp.int32)),
            jnp.array_equal(polarization, jnp.array([1], dtype=jnp.int32)),
            jnp.array_equal(polarization, jnp.array([2], dtype=jnp.int32))
        ],
        choicelist=[
            tmm_s_or_p_pol_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, jnp.array([False], dtype=bool), wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num),
            tmm_s_or_p_pol_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, jnp.array([True], dtype=bool), wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num),
            tmm_u_pol_sensitive(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, coherency_list, wl_angle_shape, key, deposition_deviation_percentage, sensitivity_optimization_sample_num)
        ]
    )


def coh_tmm_s_or_p_pol(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, polarization):
    """
    Computes coherent TMM for a given single polarization.

    Returns:
        Coherent simulation results [R, T, A].
    """
    R, T = vectorized_coh_tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, polarization)
    A = jnp.subtract(1, jnp.add(R, T))
    return jnp.array([R, T, A])


def coh_tmm_u_pol(data, material_distribution, thickness_list, wavelengths, angle_of_incidences):
    """
    Computes coherent TMM results for unpolarized light by averaging over s and p polarizations.

    Returns:
        Averaged coherent results [R, T, A].
    """
    R_s, T_s = vectorized_coh_tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, jnp.array([False], dtype=bool))
    R_p, T_p = vectorized_coh_tmm(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, jnp.array([True], dtype=bool))
    R = jnp.true_divide(R_s + R_p, 2)
    T = jnp.true_divide(T_s + T_p, 2)
    A = jnp.subtract(1, jnp.add(R, T))
    return jnp.array([R, T, A])


def coh_tmm_sparse(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, polarization):
    """
    Dispatcher for coherent TMM based on polarization: s, p, or unpolarized.

    Returns:
        Coherent TMM results [R, T, A] based on selected polarization.
    """
    return jnp.select(
        condlist=[
            jnp.array_equal(polarization, jnp.array([0], dtype=jnp.int32)),
            jnp.array_equal(polarization, jnp.array([1], dtype=jnp.int32)),
            jnp.array_equal(polarization, jnp.array([2], dtype=jnp.int32))
        ],
        choicelist=[
            coh_tmm_s_or_p_pol(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, jnp.array([False], dtype=bool)),
            coh_tmm_s_or_p_pol(data, material_distribution, thickness_list, wavelengths, angle_of_incidences, jnp.array([True], dtype=bool)),
            coh_tmm_u_pol(data, material_distribution, thickness_list, wavelengths, angle_of_incidences)
        ]
    )
