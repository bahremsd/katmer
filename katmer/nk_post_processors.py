import jax.numpy as jnp  # Import the jax numpy module for numerical and mathematical operations, used for efficient array manipulation and computations on CPUs/GPUs/TPUs.
from jax.lax import conv_general_dilated  # XLA-optimized loop in JAX, without Python's loop overhead, ensuring functional-style , conv_general_dilated
from jax import Array  # Import the Array class from jax, which is used for creating arrays in JAX, though it's included here primarily for type specification purpose.
from jax.typing import ArrayLike  # Import ArrayLike from jax.typing. It is an annotation for any value that is safe to implicitly cast to a JAX array
from typing import Callable, Literal, Tuple, Dict  # Importing type hints: Callable for functions and List for lists of elements  

# Define allowed method names
SmoothingMethod = Literal["gaussian", "moving_average", "exponential", "hamming"]


def gaussian_smooth(nk: ArrayLike,
                    window_range: Tuple[float, float] = (-10.0, 10.0),
                    kernel_size: int = 20,
                    sigma: float = 4.0,
                    amplitude: float = 0.5,
                    decay: float = 0.8) -> Array:
    """
    Applies Gaussian smoothing to the input nk array.
    """
    x = jnp.linspace(window_range[0], window_range[1], kernel_size)
    kernel = jnp.exp(-amplitude * (x / sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)
    smoothed = conv_general_dilated(
        nk.reshape(1, 1, -1),
        kernel.reshape(1, 1, -1),
        window_strides=(1,),
        padding='VALID'
    ).squeeze()
    return smoothed


def moving_average_smooth(nk: ArrayLike,
                          kernel_size: int = 10,
                          window_range: Tuple[float, float] = (-1.0, 1.0),
                          sigma: float = 1.0,
                          amplitude: float = 1.0,
                          decay: float = 0.8) -> Array:
    """
    Applies a simple moving average filter to the input nk array.
    
    Additional parameters (not used) are accepted to maintain a common function signature.
    """
    kernel = jnp.ones(kernel_size) / kernel_size
    smoothed = conv_general_dilated(
        nk.reshape(1, 1, -1),
        kernel.reshape(1, 1, -1),
        window_strides=(1,),
        padding='VALID'
    ).squeeze()
    return smoothed


def exponential_smooth(nk: ArrayLike,
                       decay: float = 0.8,
                       kernel_size: int = 50,
                       window_range: Tuple[float, float] = (-1.0, 1.0),
                       sigma: float = 1.0,
                       amplitude: float = 1.0) -> Array:
    """
    Applies exponential decay smoothing to the input nk array.
    """
    weights = jnp.array([decay**i for i in range(kernel_size)])
    weights = weights / jnp.sum(weights)
    kernel = weights[::-1]  # Reverse for causal filter
    smoothed = conv_general_dilated(
        nk.reshape(1, 1, -1),
        kernel.reshape(1, 1, -1),
        window_strides=(1,),
        padding='VALID'
    ).squeeze()
    return smoothed


def hamming_smooth(nk: ArrayLike,
                   kernel_size: int = 21,
                   window_range: Tuple[float, float] = (-1.0, 1.0),
                   sigma: float = 1.0,
                   amplitude: float = 1.0,
                   decay: float = 0.8) -> Array:
    """
    Applies Hamming window smoothing to the input nk array.
    """
    n = jnp.arange(kernel_size)
    kernel = 0.54 - 0.46 * jnp.cos(2 * jnp.pi * n / (kernel_size - 1))
    kernel = kernel / jnp.sum(kernel)
    smoothed = conv_general_dilated(
        nk.reshape(1, 1, -1),
        kernel.reshape(1, 1, -1),
        window_strides=(1,),
        padding='VALID'
    ).squeeze()
    return smoothed


# Dictionary linking method names to smoothing functions
smoothing_methods: Dict[SmoothingMethod, Callable[..., Array]] = {
    "gaussian": gaussian_smooth,
    "moving_average": moving_average_smooth,
    "exponential": exponential_smooth,
    "hamming": hamming_smooth,
}


def smooth_nk(nk: ArrayLike,
              method: SmoothingMethod = "gaussian",
              window_range: Tuple[float, float] = (-10.0, 10.0),
              sigma: float = 4.0,
              amplitude: float = 0.5,
              kernel_size: int = 20,
              decay: float = 0.8) -> Array:
    """
    Generic smoothing interface for 1D nk arrays.

    Parameters:
    - nk: 1D array of n or k values to smooth.
    - method: Smoothing method to use.
    - All other parameters are passed to the selected method function.
    
    Returns:
    - Smoothed 1D array.
    """
    if method not in smoothing_methods:
        raise ValueError(f"Unsupported smoothing method: {method}")
    
    return smoothing_methods[method](
        nk=nk,
        window_range=window_range,
        sigma=sigma,
        amplitude=amplitude,
        kernel_size=kernel_size,
        decay=decay
    )