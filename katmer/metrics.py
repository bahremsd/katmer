import jax.numpy as jnp  # Import the jax numpy module for numerical and mathematical operations, used for efficient array manipulation and computations on CPUs/GPUs/TPUs.
import equinox as eqx  # Importing Equinox, a library for neural networks and differentiable programming in JAX
from jax import Array  # Import the Array class from jax, which is used for creating arrays in JAX, though it's included here primarily for type specification purpose.
from jax.typing import ArrayLike  # Import ArrayLike from jax.typing. It is an annotation for any value that is safe to implicitly cast to a JAX array
from typing import Union

from .stack import Stack, OneUnkMaterialStack_N, OneUnkMaterialSparseStack_N, OneUnkMaterialStack_NK, OneUnkMaterialSparseStack_NK

@eqx.filter_jit
def mse_loss_thickness_optimizer(stack: Stack, 
                                 target: ArrayLike):
    
    return jnp.mean((stack() - target) ** 2)

@eqx.filter_jit
def mse_loss_nk_optimizer(one_unk_material_stack: Union[OneUnkMaterialStack_N, OneUnkMaterialSparseStack_N, OneUnkMaterialStack_NK, OneUnkMaterialSparseStack_NK],
                          thickness: ArrayLike, 
                          experimental_data: ArrayLike):
    
    prediction = one_unk_material_stack(thickness)
    return jnp.mean((prediction - experimental_data) ** 2)