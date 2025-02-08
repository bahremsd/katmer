import jax.numpy as jnp  # Import the jax numpy module for numerical and mathematical operations
from jax import Array  # Import the Array class from jax, which is used for creating arrays in JAX, though it's included here primarily for type specification purpose.
from jax.typing import ArrayLike  # Import ArrayLike from jax.typing. It is an annotation for any value that is safe to implicitly cast to a JAX array

def objective_to_function(expression: str):
    """
    Converts a mathematical expression string involving R, T, and A into a JAX-compatible function.

    Args:
        expression (str): A string representing a mathematical function of R, T, and A.
                          Example: "T", "R*T", "jnp.sin(T)".

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
            A function that takes R, T, and A as JAX arrays and evaluates the given expression.

    Explanation:
        - R (Reflection): The fraction of incident light that is reflected by the thin film.
        - T (Transmission): The fraction of incident light that passes through the thin film.
        - A (Absorption): The fraction of incident light that is absorbed within the thin film.

        By the law of energy conservation in optics: R + T + A = 1

        Examples:
        1. Only Transmission (T) Matters:
           expression = "T"  → Objective function returns T.
        
        2. Reflection and Transmission Product:
           expression = "R * T"  → Objective function returns R * T.
        
        3. Using a Nonlinear Function (Sine of Transmission):
           expression = "jnp.sin(T)"  → Objective function returns sin(T).
        
        4. More Complex Expressions:
           expression = "R**2 + jnp.exp(-T) - A"  → Objective function evaluates R^2 + e^(-T) - A.
    """
    return lambda R,T,A: eval(expression)