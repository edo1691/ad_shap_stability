import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def plot_3d_surface(df, x_label, y_label, z_label, cmap='inferno', y_step=10,
                    opt_x=None, opt_y=None, opt_z=None, opt_color='red'):
    """
    Generates a 3D surface plot for the specified columns in a pandas DataFrame and highlights an optimal point.

    Parameters:
    - df: pandas.DataFrame containing the data.
    - x_label: The name of the column to be used as the X-axis.
    - y_label: The name of the column to be used as the Y-axis.
    - z_label: The name of the column to be used as the Z-axis.
    - cmap: The colormap for the surface. Defaults to 'inferno'.
    - y_step: The increment for ticks on the Y-axis. Defaults to 10.
    - opt_x: The x-coordinate of the optimal point. Optional.
    - opt_y: The y-coordinate of the optimal point. Optional.
    - opt_z: The z-coordinate of the optimal point. Optional.
    - opt_color: The color of the optimal point. Defaults to 'red'.
    """
    # Extracting values from the DataFrame
    x = df[x_label].values
    y = df[y_label].values
    z = df[z_label].values

    # Creating grid data, starting from 0 for both x and y
    xi = np.linspace(0, x.max(), 100)
    yi = np.linspace(0, y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolating z values on the grid
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # Creating figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting surface plot
    surf = ax.plot_surface(xi, yi, zi, cmap=cmap, edgecolor='none')

    # Highlighting the optimal point if provided
    if opt_x is not None and opt_y is not None and opt_z is not None:
        ax.scatter(opt_x, opt_y, opt_z, color=opt_color, s=50, depthshade=True)

    # Adding labels and adjusting axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_xlim([0, x.max()])
    ax.set_ylim([0, y.max()])
    ax.set_xticks(np.arange(0, x.max(), max(x.max() / 6, 1)))
    ax.set_yticks(np.arange(0, y.max() + y_step, y_step))

    # Adding a color bar
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)

    plt.show()
