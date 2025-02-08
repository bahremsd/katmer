import jax.numpy as jnp  # Import JAX's version of NumPy
from jax.lax import scan  # Import the `scan` function, which allows efficient looping and state updates in jax.ndarray computations
from jax import Array  # Import the `Array` type from JAX for precise type hinting with JAX arrays
from jax.typing import ArrayLike  # Import `ArrayLike` for type hinting, accommodating various array-like structures compatible with JAX

def compute_first_layer_matrix_coherent(r: ArrayLike, t: ArrayLike) -> Array:
    """
    This function calculates the transfer matrix for the interface between the incoming and the first layer of the
    multilayer coating. This function assumes the first layer is coherent and should not be used 
    if the first layer is not coherent.

    Arguments:
    - r (ArrayLike): The reflection coefficient at the interface between the incoming medium and the first layer of the thin film
    - t (ArrayLike): The transmission coefficient at the interface between the incoming medium and the first layer of the thin film

    Returns:
    - first_layer_matrix (Array): A 2x2 matrix representing the transfer matrix for the first layer at the interface 
    between the incoming and outgoing medium, adjusted by the transmission coefficient t.
    """
    # Create a 2x2 matrix representing the reflection and transmission coefficients
    # The matrix structure is [[1, r], [r, 1]], where r is the reflection coefficient.
    first_layer_matrix = jnp.array([[1, r],
                                    [r, 1]])

    # Scale the matrix by dividing each element by the transmission coefficient t.
    # This ensures proper normalization of the transfer matrix with respect to t.
    first_layer_matrix = jnp.multiply(jnp.true_divide(1, t), first_layer_matrix)

    # Return the adjusted transfer matrix for the first layer.
    # This matrix describes the interface between the incoming and outgoing media.
    return first_layer_matrix

def compute_first_layer_matrix_incoherent(r_forward: ArrayLike, t_forward: ArrayLike, 
                                          r_backward: ArrayLike, t_backward: ArrayLike):
    """
    This function calculates the transfer matrix for the interface between the incoming and the first layer of the
    multilayer coating where the first layer is incoherent.

    Arguments:
    - r_forward (ArrayLike): The reflection coefficient for the forward direction at the interface between 
    the incoming medium and the first layer.
    - t_forward (ArrayLike): The transmission coefficient for the forward direction at the interface between 
    the incoming medium and the first layer.
    - r_backward (ArrayLike): The reflection coefficient for the backward direction at the interface between 
    the incoming medium and the first layer.
    - t_backward (ArrayLike): The transmission coefficient for the backward direction at the interface between 
    the incoming medium and the first layer.

    Returns:
    - first_layer_matrix (Array): A 2x2 matrix representing the transfer matrix for the first layer at the interface 
    between the incoming medium and the first layer, adjusted by the forward transmission and reflection coefficients
    """
    # Initialize the element at position (0, 0) of the transfer matrix
    transfer_matrix_00 = 1
    
    # Calculate the element at position (0, 1) of the transfer matrix using the backward reflection coefficient
    transfer_matrix_01 = jnp.multiply(-1, r_backward)
    
    # Calculate the element at position (1, 0) of the transfer matrix using the forward reflection coefficient
    transfer_matrix_10 = r_forward
    
    # Calculate the element at position (1, 1) using the forward and backward transmission coefficients,
    # and the forward and backward reflection coefficients
    transfer_matrix_11 = jnp.subtract(jnp.multiply(t_forward, t_backward), jnp.multiply(r_forward, r_backward))

    # Construct the 2x2 transfer matrix from the calculated elements
    first_layer_matrix = jnp.array([[transfer_matrix_00, transfer_matrix_01],
                                    [transfer_matrix_10, transfer_matrix_11]])

    # Scale the entire matrix by dividing each element by the forward transmission coefficient t_forward
    first_layer_matrix = jnp.multiply(jnp.true_divide(1, t_forward), first_layer_matrix)

    # Return the adjusted transfer matrix for the first layer
    return first_layer_matrix

def coh_matmul(carry: ArrayLike, phase_r_t: ArrayLike):
    """
    This function performs a single step of matrix multiplication in the transfer matrix method for coherent thin films.
    It constructs the transfer matrix M_i using the given phase, reflection, and transmission coefficients, and then multiplies 
    it with the accumulated matrix carry (representing M_carry). This is part of the iterative process of computing the overall 
    transfer matrix M = M_1 @ M_2 @ ... @ M_N in a computationally efficient manner using JAX.

    Arguments:
    carry: ArrayLike
        The matrix that accumulates the multiplication process in the transfer matrix method. 
        It represents the product of all previous transfer matrices (M_carry in the method).
    phase_r_t: ArrayLike
        A 1D array containing three elements:
        - phase_r_t[0]: A phase term used to compute the exponential factors in the transfer matrix. 
          This is a real number representing the phase in radians.
        - phase_r_t[1]: A reflection coefficient (r). It represents the ratio of reflected wave amplitude 
          to the incident wave amplitude at the interface.
        - phase_r_t[2]: A transmission coefficient (t). It represents the ratio of transmitted wave amplitude 
          to the incident wave amplitude at the interface.

    Returns:
    result: tuple
        A tuple containing two identical elements:
        - The updated matrix after multiplying the current transfer matrix (M_i) with the accumulated matrix (M_carry).
        - A placeholder for `jax.lax.scan`, which requires a tuple to function correctly.
    """

    # Compute the (0,0) element of the transfer matrix: exp(-i * phase) / t
    transfer_matrix_00 = jnp.exp(jnp.multiply(jnp.array([-1j], dtype = jnp.complex64), phase_r_t.at[0].get()))

    # Compute the (1,1) element of the transfer matrix: exp(i * phase) / t
    transfer_matrix_11 = jnp.exp(jnp.multiply(jnp.array([1j], dtype = jnp.complex64), phase_r_t.at[0].get()))

    # Compute the (0,1) element of the transfer matrix: r * exp(-i * phase) / t
    transfer_matrix_01 = jnp.multiply(phase_r_t.at[1].get(), transfer_matrix_00)

    # Compute the (1,0) element of the transfer matrix: r * exp(i * phase) / t
    transfer_matrix_10 = jnp.multiply(phase_r_t.at[1].get(), transfer_matrix_11)

    # Assemble the full 2x2 transfer matrix M_i
    # M_i = (1/t) * [[exp(-i * phase), r * exp(-i * phase)],
    #                [r * exp(i * phase), exp(i * phase)]]
    transfer_matrix = jnp.multiply(jnp.true_divide(1, phase_r_t.at[2].get()), jnp.array([[transfer_matrix_00, transfer_matrix_01],
                                                                                         [transfer_matrix_10, transfer_matrix_11]]))
    
    # Multiply the accumulated matrix (carry) with the current transfer matrix (M_i)
    result = jnp.squeeze(jnp.matmul(carry, transfer_matrix))

    # Return the updated matrix and a placeholder (same matrix) for jax.lax.scan compatibility
    return result, result

def coh_cascaded_matrix_multiplication(phases: ArrayLike, rts: ArrayLike) -> Array:
    """
    This function calculates the total transfer matrix of a multilayer thin-film structure by performing a cascaded matrix multiplication. 
    It takes in phase shifts and reflection/transmission coefficients for each layer and computes the overall transfer coefficient matrix. 
    Arguments:
        phases (ArrayLike): 
            A 1D array of size N (number of layers) representing the accumulated phase shift in each layer of the multilayer structure. 
            Each entry corresponds to the phase shift of the wave as it passes through the respective layer.

        rts (ArrayLike): 
            A 2D array of size [N, 2], where N is the number of layers. 
            For the i-th layer:
                rts[i, 0] represents the reflection coefficient `r` for the i+1-th layer.
                rts[i, 1] represents the transmission coefficient `t` for the i+1-th layer.
                rts[-1, :] represents the transmission and reflection coefficients of the Nth layer and outgoing medium interface.

    Returns:
        Array: 
            A 2x2 complex matrix representing the total transfer matrix of the multilayer system (except incoming r matrix). 
            This matrix encapsulates the overall effect of all the layers in terms of phase shifts, reflection, and transmission.
    """
    # Concatenate the phases (1D array) with the r and t coefficients (2D array) along a new axis
    # This creates a 2D array where each row represents a layer, and each row contains:
    # [phase, r, t] values for that layer.
    phase_rt_stack = jnp.concat([jnp.expand_dims(phases, 1), rts], axis=1)

    # Initialize the result as a 2x2 identity matrix of complex numbers
    # The identity matrix serves as the neutral element for matrix multiplication.
    initial_value = jnp.eye(2, dtype=jnp.complex64)

    # Use the `lax.scan` function to iteratively multiply matrices over the layers
    # `coh_matmul` is a user-defined function (not shown here) that performs the matrix multiplication for a single layer.
    # `initial_value` is the starting matrix, and `phase_rt_stack` provides the data for each step.
    # The scan function returns the final accumulated matrix in `result`.
    result, _ = scan(coh_matmul, initial_value, phase_rt_stack)  # Scan function accumulates results of coh_matmul over the matrices.

    # Return the resulting 2x2 matrix, representing the total transfer matrix for the multilayer structure.
    return result

def incoh_matmul(carry: ArrayLike, pass_rf_tf_rb_tb: ArrayLike):
    """ 
    This function performs a single step of matrix multiplication in the transfer matrix method for coherent thin films.
    It constructs the transfer matrix M_i using the given phase, reflection, and transmission coefficients, and then multiplies 
    it with the accumulated matrix carry (representing M_carry). This is part of the iterative process of computing the overall 
    transfer matrix M = M_1 @ M_2 @ ... @ M_N in a computationally efficient manner using JAX.

    Arguments:
    - carry (ArrayLike): A 2x2 matrix representing the accumulated transfer matrix from previous layers.
    - pass_rf_tf_rb_tb (ArrayLike): A 1D array containing reflection, transmission, and passing ratio coefficients 
    required to construct the transfer matrix:
    * pass_rf_tf_rb_tb[0]: Passing ratio
    * pass_rf_tf_rb_tb[1]: Forward reflection coefficient (r_f)
    * pass_rf_tf_rb_tb[2]: Forward transmission coefficient (t_f)
    * pass_rf_tf_rb_tb[3]: Backward reflection coefficient (r_b)
    * pass_rf_tf_rb_tb[4]: Backward transmission coefficient (t_b)

    Returns:
    - result (ArrayLike): Updated accumulated transfer matrix after multiplication.
    - result (ArrayLike): The same updated transfer matrix is returned twice for JAX compatibility.
    """

    # Compute the first element of the transfer matrix: 1 / passing ratio
    transfer_matrix_00 = jnp.true_divide(1, pass_rf_tf_rb_tb.at[0].get())

    # Compute the second element of the transfer matrix: -r_b / passing ratio
    transfer_matrix_01 = jnp.true_divide(jnp.multiply(-1, pass_rf_tf_rb_tb.at[3].get()), pass_rf_tf_rb_tb.at[0].get())

    # Compute the third element of the transfer matrix: passing ratio * r_f
    transfer_matrix_10 = jnp.multiply(pass_rf_tf_rb_tb.at[0].get(), pass_rf_tf_rb_tb.at[1].get())

    # Compute the fourth element of the transfer matrix
    # passing ratio * (t_f * t_b - r_f * r_b)
    transfer_matrix_11 = jnp.multiply(pass_rf_tf_rb_tb.at[0].get(), 
                                      jnp.subtract(jnp.multiply(pass_rf_tf_rb_tb.at[2].get(), pass_rf_tf_rb_tb.at[4].get()), 
                                                   jnp.multiply(pass_rf_tf_rb_tb.at[1].get(), pass_rf_tf_rb_tb.at[3].get())))

    # Construct the full transfer matrix M_i
    transfer_matrix = jnp.multiply(jnp.true_divide(1, pass_rf_tf_rb_tb.at[2].get()), 
                                   jnp.array([[transfer_matrix_00, transfer_matrix_01], 
                                              [transfer_matrix_10, transfer_matrix_11]]))

    # Multiply the accumulated transfer matrix with the current layer's transfer matrix
    result = jnp.matmul(carry, transfer_matrix)

    # Remove unnecessary dimensions if applicable
    result = jnp.squeeze(result)

    # Return the updated transfer matrix twice (likely for JAX compatibility in a scan function)
    return result, result

def incoh_cascaded_matrix_multiplication(forward_magnitudes: ArrayLike, backward_magnitudes: ArrayLike, layer_passes: ArrayLike) -> Array:
    """
    This function performs cascaded matrix multiplication for an incoherent optical system using JAX.
    It iterates through layer-wise power distributions and magnitudes, accumulating their effects via matrix multiplication.

    Arguments:
    forward_magnitudes (ArrayLike):
        A 2D array of shape [N, 2] where N is the number of layers.
        Each row contains the forward-traveling field magnitudes for the corresponding layer.

    backward_magnitudes (ArrayLike):
        A 2D array of shape [N, 2] where N is the number of layers.
        Each row contains the backward-traveling field magnitudes for the corresponding layer.

    layer_passes (ArrayLike):
        A 1D array of length N representing the passing fractions associated with each layer.
        These values are the  fraction of light that successfully passes through the incoherent layers

    Returns:
    Array:
        A 2x2 matrix representing the accumulated effect of the layers in an incoherent system.
        The resulting matrix encodes the system's overall transmission and reflection characteristics.
    """
    # Stack the layer power distribution, forward magnitudes, and backward magnitudes into a single array
    pass_rf_tf_rb_tb_stack = jnp.concat([jnp.expand_dims(layer_passes, 1), forward_magnitudes, backward_magnitudes], axis=1)

    # Initialize the matrix as a 2x2 identity matrix (represents no transformation initially)
    initial_value = jnp.eye(2, dtype=jnp.float32)

    # Perform a scan (folding operation) over layers, applying matmul_incoherent iteratively
    result, _ = scan(incoh_matmul, initial_value, pass_rf_tf_rb_tb_stack)

    # Return the final accumulated matrix, representing the total effect of the multilayer system
    return result
