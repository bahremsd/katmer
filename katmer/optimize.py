
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, Float, Int, PyTree
from typing import Union

def optimize_thickness(optimizer, loss_func, stack, target, num_of_iter, save_log = True):

    opt_state = optimizer.init(eqx.filter(stack, eqx.is_array))

    min_thickness_in_um = stack.min_thickness
    max_thickness_in_um = stack.max_thickness

    def clamp_values(model: Stack):
        """Clamp the refractive index values within the specified bounds."""
        clamped_model = eqx.tree_at(
            lambda x: x.refractive_index,  # Target refractive indices attribute
            model,
            replace_fn=lambda x: jnp.clip(x, min_thickness_in_um, min_thickness_in_um),
        )
        return clamped_model


    @eqx.filter_jit
    def tmm_step(
        model: Stack,
        opt_state: PyTree,
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_func)(model, target)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        model = clamp_values(model)
        return model, opt_state, loss_value

    stack, opt_state, _ = tmm_step(stack, opt_state)

    iter_vs_loss = []

    for i in range(num_of_iter):
        stack, opt_state, state_loss = tmm_step(stack, opt_state)
        iter_vs_loss.append(state_loss)

    return stack, iter_vs_loss


def optimize_refractive_index(optimizer, loss_func, stack, experimental_data, experimental_thicknesses, num_of_iter, save_log = True):

    opt_state = optimizer.init(eqx.filter(stack, eqx.is_array))

    min_refractive_index = stack.min_refractive_index
    max_refractive_index = stack.max_refractive_index
    min_extinction_coefficient = stack.min_extinction_coeff
    max_extinction_coefficient = stack.max_extinction_coeff
    dataset_sample_size = len(experimental_data)

    def clamp_values(model):
        """Clamp the refractive index values within the specified bounds."""
        clamped_model = eqx.tree_at(
            lambda x: x.refractive_index,  # Target refractive indices attribute
            model,
            replace_fn=lambda x: jnp.clip(x, min_refractive_index, max_refractive_index),
        )
        clamped_model = eqx.tree_at(
            lambda x: x.extinction_coefficient,  # Target refractive indices attribute
            clamped_model,
            replace_fn=lambda x: jnp.clip(x, min_extinction_coefficient, max_extinction_coefficient),
        )
        return clamped_model



    @eqx.filter_jit
    def tmm_step(
        model: Stack,
        opt_state: PyTree,
        data_idx: int,
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_func)(model, experimental_thicknesses.at[data_idx].get(), experimental_data.at[data_idx].get())
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        model = clamp_values(model)
        return model, opt_state, loss_value

    stack, opt_state, _ = tmm_step(stack, opt_state, 0)

    iter_vs_loss = []

    for i in range(num_of_iter):
      loss = 0
      for sample_idx in range(dataset_sample_size):
        stack, opt_state, state_loss = tmm_step(stack, opt_state, sample_idx)
        loss += state_loss
      #plt.plot(stack(experimental_thicknesses.at[1].get()))
      #plt.plot(experimental_data.at[1].get())
      #plt.show()
      #plt.close("all")
      iter_vs_loss.append(loss/dataset_sample_size)


    return stack, iter_vs_loss