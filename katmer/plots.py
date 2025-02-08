import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Union, List, Optional
import matplotlib.cm as cm

def plot_material_and_thicknesses(
        _thicknesses: Union[np.ndarray, jnp.ndarray], 
        _material_distribution: List[str], 
        _material_set: List[str], 
        filename: Optional[str] = None) -> None:
    """
    This function creates a plot that visualizes the material distribution and corresponding 
    layer thicknesses in a multilayer thin film structure.

    Arguments:
    _thicknesses : numpy array or jax array 
        The thicknesses of each layer in the structure.
    _material_distribution : list of strings 
        Indicates which material is in each layer.
    _material_set : list of strings 
        The set of materials that can be used during optimization.
    filename : str or None 
        The name of the file to save the plot. If None, the plot is displayed instead.

    The function does not return anything. It either shows the plot or saves it as a PNG file.
    """

    # Determine the color map based on the material set
    color_map = plt.get_cmap('Set2', len(_material_set))
    color_dict = {material: color_map(i) for i, material in enumerate(_material_set)}

    # Normalize thicknesses for the plot
    total_thickness = np.sum(_thicknesses)
    normalized_thicknesses = _thicknesses / total_thickness

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))  # More compact figure size
    
    # Initial y-offset for stacking layers
    y_offset = 0
    
    # Plot each layer
    for i, (material, thickness) in enumerate(zip(_material_distribution, normalized_thicknesses)):
        ax.bar(0, thickness, bottom=y_offset, color=color_dict[material], edgecolor='black', width=2, label=material)
        y_offset += thickness

    # Set the x-ticks and y-ticks to be invisible
    ax.set_xticks([])
    ax.set_yticks([])

    # Add the layer thicknesses and material labels
    y_offset = 0  # Reset y_offset for text placement
    for i, (material, thickness, norm_thickness) in enumerate(zip(_material_distribution, _thicknesses, normalized_thicknesses)):
        # Check if the normalized thickness is greater than 5% (0.05) before adding text
        if norm_thickness > 0.05:
            mid_y = y_offset + norm_thickness / 2
            ax.text(0.5, mid_y, f'{thickness:.0f} nm', 
                    ha='center', va='center', color='black', fontsize=7, fontweight='bold')
        y_offset += norm_thickness

    # Set the plot limits and labels
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1.0)  # Adjust x-axis to prevent text overlap
    ax.set_title('Material Distribution and Thicknesses', fontsize=16, fontweight='bold')
    #ax.set_ylabel('Normalized Thickness', fontsize=12)  # Add y-axis label for normalized thickness

    # Remove legend duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(), loc='center left', fontsize=12, bbox_to_anchor=(1, 0.5),
                       title="Materials", title_fontsize='13', frameon=True, shadow=True, fancybox=True)
    
    # Customize the legend appearance
    plt.setp(legend.get_title(), fontsize=14, fontweight='bold')  # Make the title of the legend larger and bold
    legend.get_frame().set_edgecolor('black')  # Set the legend frame color
    legend.get_frame().set_linewidth(1.5)  # Increase the legend frame line width

    # Adjust the plot layout with added space for the legend
    plt.subplots_adjust(right=0.7)  # Increase space between the plot and the legend


    # Save or show the plot
    if filename is None:
        plt.show()
    else:
        plt.savefig(f"{filename}.png", dpi=300)
        

def plot_layer_distribution(
    _thicknesses: Union[np.ndarray, jnp.ndarray],
    _distribution: Union[np.ndarray, jnp.ndarray],
    filename: Optional[str] = None
) -> None:
    """
    Plots the distribution of energy or absorption across the layers of a multilayer thin film structure.

    Args:
        _thicknesses (Union[np.ndarray, jnp.ndarray]): An array of layer thicknesses for the multilayer thin film. 
                                                      The length must match the length of _distribution.
        _distribution (Union[np.ndarray, jnp.ndarray]): An array of energy or absorption distribution values corresponding
                                                        to each layer in the multilayer thin film. The length must
                                                        match the length of _thicknesses.
        filename (Optional[str]): If provided, the plot is saved to this filename with a .png extension. If None, the
                                  plot is displayed but not saved.

    Returns:
        None: The function either saves the plot to a file or displays it but does not return any value.
    """

    # Convert inputs to numpy arrays for compatibility with matplotlib
    if isinstance(_thicknesses, jnp.ndarray):
        _thicknesses = np.array(_thicknesses)
    if isinstance(_distribution, jnp.ndarray):
        _distribution = np.array(_distribution)
    
    # Validate input lengths
    if len(_thicknesses) != len(_distribution):
        raise ValueError("The length of _thicknesses and _distribution must be equal.")
    
    # Normalize the thicknesses and distribution for plotting
    norm = plt.Normalize(vmin=_distribution.min(), vmax=_distribution.max())
    cmap = cm.get_cmap('hot')  # Use the 'viridis' colormap, can be changed as needed
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate cumulative thicknesses for layer plotting
    cum_thicknesses = np.concatenate(([0], np.cumsum(_thicknesses)))
    
    # Plot each layer with the corresponding color based on the distribution
    for i in range(len(_thicknesses)):
        ax.fill_betweenx(
            [cum_thicknesses[i], cum_thicknesses[i + 1]],
            0, 1, 
            color=cmap(norm(_distribution[i])),
            edgecolor='black',  # Black strip around each layer
            linewidth=1        # Width of the black border
        )
    
    ax.set_title('Field Distribution in Multilayer Thin Film', fontsize=16, fontweight='bold')

    # Add colorbar to show distribution scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Distribution Value')
    
    # Remove x and y labels, and frame around the plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Save or show the plot
    if filename:
        plt.savefig(f"{filename}.png", bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    
    plt.close(fig)