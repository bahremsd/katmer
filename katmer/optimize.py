import jax.numpy as jnp
import equinox as eqx
from jax.typing import ArrayLike
from typing import Union, Callable, List, Tuple

from .stack import (
    Stack,
    OneUnkMaterialStack_N,
    OneUnkMaterialStack_NK,
    OneUnkMaterialSparseStack_N,
    OneUnkMaterialSparseStack_NK,
)

def optimize_thickness(
    optimizer,
    loss_func: Callable,
    stack: Stack,
    target: ArrayLike,
    num_of_iter: int,
    save_log: bool = True
) -> Tuple[Stack, List[float]]:
    """
    Optimize the thickness distribution in a multilayer stack to match a target spectrum.

    Args:
        optimizer: An Optax optimizer object for gradient updates.
        loss_func: A callable that computes loss between predicted and target spectrum.
        stack: A `Stack` object with fixed materials and trainable thickness values.
        target: Target spectrum (transmission/reflection/absorption) to fit.
        num_of_iter: Number of optimization iterations.
        save_log: Flag to save log (unused but kept for interface consistency).

    Returns:
        Updated stack with optimized thickness and list of loss values per iteration.
    """

    opt_state = optimizer.init(eqx.filter(stack, eqx.is_array))

    min_thickness_in_um = stack.min_thickness
    max_thickness_in_um = stack.max_thickness

    def clamp_values(model: Stack) -> Stack:
        """Clamp the layer thicknesses within defined min/max values."""
        clamped_model = eqx.tree_at(
            lambda x: x.refractive_index,  # Assuming thickness is represented here
            model,
            replace_fn=lambda x: jnp.clip(x, min_thickness_in_um, max_thickness_in_um),
        )
        return clamped_model

    @eqx.filter_jit
    def tmm_step(
        model,
        opt_state):

        """Single optimization step for updating thickness."""
        loss_value, grads = eqx.filter_value_and_grad(loss_func)(model, target)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        model = clamp_values(model)
        return model, opt_state, loss_value

    stack, opt_state, _ = tmm_step(stack, opt_state)

    iter_vs_loss: List[float] = []

    for _ in range(num_of_iter):
        stack, opt_state, state_loss = tmm_step(stack, opt_state)
        iter_vs_loss.append(state_loss)

    return stack, iter_vs_loss


def optimize_refractive_index(
    optimizer,
    loss_func: Callable,
    stack: Union[OneUnkMaterialStack_N, OneUnkMaterialStack_NK],
    experimental_data: ArrayLike,
    experimental_thicknesses: ArrayLike,
    num_of_iter: int,
    save_log: bool = True
) -> Tuple[Union[OneUnkMaterialStack_N, OneUnkMaterialStack_NK], List[float]]:
    """
    Optimize the refractive index (n) and optionally the extinction coefficient (k)
    to fit experimental spectral data (T/R/A) for fixed material distribution.

    Args:
        optimizer: Optax optimizer for gradient updates.
        loss_func: Callable to compute loss for spectrum prediction.
        stack: Stack with one unknown material (n or nk).
        experimental_data: Array of measured transmission/reflection/absorption data.
        experimental_thicknesses: Layer thickness values for each experiment.
        num_of_iter: Number of full passes through the dataset.
        save_log: Flag to save logs (not currently used).

    Returns:
        Tuple of optimized stack and a list of mean losses per iteration.
    """

    opt_state = optimizer.init(eqx.filter(stack, eqx.is_array))

    # Clipping bounds for optimization
    min_refractive_index = stack.min_refractive_index
    max_refractive_index = stack.max_refractive_index
    min_extinction_coefficient = stack.min_extinction_coeff
    max_extinction_coefficient = stack.max_extinction_coeff
    dataset_sample_size = len(experimental_data)

    def clamp_values(model: Union[OneUnkMaterialStack_N, 
                                  OneUnkMaterialStack_NK]) -> Union[OneUnkMaterialStack_N, 
                                                                    OneUnkMaterialStack_NK]:
        """Clip optimized values of n and k within physical bounds."""
        clamped_model = eqx.tree_at(
            lambda x: x.refractive_index,
            model,
            replace_fn=lambda x: jnp.clip(x, min_refractive_index, max_refractive_index),
        )
        clamped_model = eqx.tree_at(
            lambda x: x.extinction_coefficient,
            clamped_model,
            replace_fn=lambda x: jnp.clip(x, min_extinction_coefficient, max_extinction_coefficient),
        )
        return clamped_model

    @eqx.filter_jit
    def tmm_step(
        model,
        opt_state,
        data_idx):

        """Per-sample optimization step for refractive index (n/k)."""
        loss_value, grads = eqx.filter_value_and_grad(loss_func)(
            model,
            experimental_thicknesses.at[data_idx].get(),
            experimental_data.at[data_idx].get(),
        )
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        model = clamp_values(model)
        return model, opt_state, loss_value

    # Warmup step
    stack, opt_state, _ = tmm_step(stack, opt_state, 0)

    iter_vs_loss: List[float] = []

    for _ in range(num_of_iter):
        loss = 0.0
        for sample_idx in range(dataset_sample_size):
            stack, opt_state, state_loss = tmm_step(stack, opt_state, sample_idx)
            loss += state_loss
        iter_vs_loss.append(loss / dataset_sample_size)

    return stack, iter_vs_loss


def optimize_refractive_index_sparse_stacks(
    optimizer,
    loss_func: Callable,
    stack: Union[OneUnkMaterialSparseStack_N, OneUnkMaterialSparseStack_NK],
    experimental_data: ArrayLike,
    experimental_thicknesses: ArrayLike,
    num_of_iter: int,
    save_log: bool = True
) -> Tuple[Stack, List[float]]:
    """
    Optimize refractive index, extinction coefficient, and compression ratio
    in sparse thin film stacks (e.g., porous films with WO_{3-x}).

    Args:
        optimizer: Optax optimizer object.
        loss_func: Loss function used to compare predicted and measured data.
        stack: Stack representing a sparse film with unknown optical properties.
        experimental_data: Experimental T/R/A data array.
        experimental_thicknesses: Corresponding thickness data for each sample.
        num_of_iter: Number of optimization iterations.
        save_log: Logging flag (not used in this implementation).

    Returns:
        Optimized stack and list of average loss per iteration.
    """

    opt_state = optimizer.init(eqx.filter(stack, eqx.is_array))

    # Bounds for all optimizable parameters
    min_refractive_index = stack.min_refractive_index
    max_refractive_index = stack.max_refractive_index
    min_extinction_coefficient = stack.min_extinction_coeff
    max_extinction_coefficient = stack.max_extinction_coeff
    min_compression_ratio = stack.min_compression_ratio
    max_compression_ratio = stack.max_compression_ratio
    dataset_sample_size = len(experimental_data)

    def clamp_values(model: Union[OneUnkMaterialSparseStack_N, 
                                  OneUnkMaterialSparseStack_NK]) -> Union[OneUnkMaterialSparseStack_N, 
                                                                          OneUnkMaterialSparseStack_NK]:
        """Clamp optimized n, k, and compression ratio within physical bounds."""
        clamped_model = eqx.tree_at(
            lambda x: x.refractive_index,
            model,
            replace_fn=lambda x: jnp.clip(x, min_refractive_index, max_refractive_index),
        )
        clamped_model = eqx.tree_at(
            lambda x: x.extinction_coefficient,
            clamped_model,
            replace_fn=lambda x: jnp.clip(x, min_extinction_coefficient, max_extinction_coefficient),
        )
        clamped_model = eqx.tree_at(
            lambda x: x.compression_ratio,
            clamped_model,
            replace_fn=lambda x: jnp.clip(x, min_compression_ratio, max_compression_ratio),
        )
        return clamped_model

    @eqx.filter_jit
    def tmm_step(
        model,
        opt_state,
        data_idx):
        
        """Optimization step for sparse film stack per sample."""
        loss_value, grads = eqx.filter_value_and_grad(loss_func)(
            model,
            experimental_thicknesses.at[data_idx].get(),
            experimental_data.at[data_idx].get(),
        )
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        model = clamp_values(model)
        return model, opt_state, loss_value

    # Initial update
    stack, opt_state, _ = tmm_step(stack, opt_state, 0)

    iter_vs_loss: List[float] = []

    for _ in range(num_of_iter):
        loss = 0.0
        for sample_idx in range(dataset_sample_size):
            stack, opt_state, state_loss = tmm_step(stack, opt_state, sample_idx)
            loss += state_loss
        iter_vs_loss.append(loss / dataset_sample_size)

    return stack, iter_vs_loss