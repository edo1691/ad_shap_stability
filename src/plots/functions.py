import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_3d_surface(df, x_label, y_label, z_label, ax, fontsize_title=12, fontsize_axes=8, cmap='inferno', x_step=10, y_step=10,
                    opt_x=None, opt_y=None, opt_color='red', title='', alpha=0.5, edgecolor='0.5'):
    """
    Generates a 3D surface plot for the specified columns in a pandas DataFrame on a given Axes object,
    ensuring both X and Y axes start from the minimum to the maximum values.

    Parameters:
    - df: pandas.DataFrame containing the data.
    - x_label: The name of the column to be used as the X-axis.
    - y_label: The name of the column to be used as the Y-axis.
    - z_label: The name of the column to be used as the Z-axis.
    - ax: The Axes object on which to draw the plot.
    - cmap: The colormap for the surface. Defaults to 'inferno'.
    - x_step: The increment for ticks on the X-axis. Defaults to 10.
    - y_step: The increment for ticks on the Y-axis. Defaults to 10.
    - opt_x: The x-coordinate of the optimal point. Optional.
    - opt_y: The y-coordinate of the optimal point. Optional.
    - opt_color: The color of the vertical line. Defaults to 'red'.
    """
    x = df[x_label].values
    y = df[y_label].values
    z = df[z_label].values

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')

    surf = ax.plot_surface(xi, yi, zi, cmap=cmap, edgecolor=edgecolor, alpha=alpha)

    if opt_x is not None and opt_y is not None:
        ax.plot([opt_x, opt_x], [opt_y, opt_y], [0, z.max()], color=opt_color, linewidth=2)

    # Set grid line colors with transparency
    gridline_color = (0.5, 0.5, 0.5, 0.5)  # RGBA tuple for semi-transparent grey
    ax.xaxis._axinfo["grid"].update(color=gridline_color)
    ax.yaxis._axinfo["grid"].update(color=gridline_color)
    ax.zaxis._axinfo["grid"].update(color=gridline_color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # Set the title for the plot
    ax.set_xlabel('Number of estimators', fontsize=fontsize_axes)
    ax.set_ylabel('Features selected', fontsize=fontsize_axes)
    ax.set_zlabel(f'{z_label}', fontsize=fontsize_axes)
    ax.set_title(title, fontsize=fontsize_title)

    # Explicitly setting the limits to ensure axes start from their minimum to maximum values
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])

    # Setting the ticks for X and Y axes
    ax.set_xticks(np.arange(0, x.max() + x_step, x_step))
    ax.set_yticks(np.arange(0, y.max() + y_step, y_step))

    ax.tick_params(axis='x', labelsize=fontsize_axes)
    ax.tick_params(axis='y', labelsize=fontsize_axes)
    ax.tick_params(axis='z', labelsize=fontsize_axes)

    # Optional: Invert X axis if needed
    ax.invert_xaxis()


def plot_2d_surface(df, ax, fontsize_title=8, fontsize_axes=8, feat='precision'):
    """
    Plots precision and stability index against number of estimators for different n_feats values side by side.

    Parameters:
    - df: DataFrame containing the columns 'n_estimators', 'n_feats', 'precision', and 'stability index'.
    """
    # Assuming the function is modified to plot only one of Precision or Stability Index at a time
    # If you need to plot both in different axes, you'd call this function twice with different `ax` each time

    # Example for plotting Precision
    for n_feats in df['n_feats'].unique():
        subset = df[df['n_feats'] == n_feats]
        ax.plot(subset['n_estimators'], subset[feat], label=f'n_feats={n_feats}')  # Use ax directly

    y_ticks = [i * 0.2 for i in range(0, 6)]  # Creates a list [0, 0.1, 0.2, ..., 1.0]

    ax.set_xlabel('number of estimators', fontsize=fontsize_axes)
    ax.set_ylabel(f'{feat}', fontsize=fontsize_axes)
    ax.set_title(f'{feat} by number of estimators', fontsize=fontsize_title)
    ax.tick_params(axis='x', labelsize=fontsize_axes)
    ax.tick_params(axis='y', labelsize=fontsize_axes)
    ax.set_yticks(y_ticks)  # Set specific y-axis ticks
    ax.legend(title='# feature selected', title_fontsize=fontsize_axes, fontsize=fontsize_axes - 2, loc='lower right',
              ncol=3)


def boxplot_stability(df, size=1):
    """
    Plots box plots for the 'shap_stab' column from a single DataFrame, differentiated by the 'dataset' column,
    using grayscale colors, with added transparency to the box plots.

    Parameters:
    - df: A pandas DataFrame containing the 'shap_stab' and 'dataset' columns.
    """
    df_names = df['dataset'].unique()  # Get unique dataset names
    fig, ax = plt.subplots(figsize=(size * 1.8 * len(df_names), 6))

    # Grayscale colors for the box plots with transparency
    colors = ['0.2', '0.7']  # Dark gray for 'Benchmark', light gray for 'Our Model'
    alpha_value = 0.6  # Adjust transparency here

    positions = []
    for i, dataset_name in enumerate(df_names):
        dataset_df = df[df['dataset'] == dataset_name]

        # Assuming there are only two types for benchmark and model within each dataset
        benchmark_data = dataset_df[dataset_df['hpo'] == 'Benchmark']['stab_shap']
        model_data = dataset_df[dataset_df['hpo'] == 'Our model']['stab_shap']

        # Boxplot for benchmark data
        bp1 = ax.boxplot(benchmark_data, positions=[2 * i + 1], widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor=colors[0], alpha=alpha_value), medianprops=dict(color='black'))

        # Boxplot for model data
        bp2 = ax.boxplot(model_data, positions=[2 * i + 2], widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor=colors[1], alpha=alpha_value), medianprops=dict(color='black'))

        positions.extend([2 * i + 1, 2 * i + 2])

    ax.set_xticks([pos for pos in positions if pos % 2 == 0])
    ax.set_xticklabels(df_names)

    ax.set_ylabel('Stability Index')

    # Adjust legend to indicate transparency
    legend_patches = [Patch(color=colors[0], label='Benchmark', alpha=alpha_value),
                      Patch(color=colors[1], label='Our Model', alpha=alpha_value)]
    ax.legend(handles=legend_patches, loc='lower right')

    ax.set_xlim(0, 2 * len(df_names) + 1)

    # Flatten the list of lists into a single list
    all_values = [item for sublist in df.stab_shap for item in sublist]
    # Find the minimum value
    min_value = min(all_values)
    max_value = max(all_values)
    ax.set_ylim(min_value - 0.1, max_value + 0.05)

    plt.tight_layout()
    plt.show()


def lineplot_stability(df, ax, fontsize_title=8, fontsize_axes=6, primary_feat='precision', secondary_feat='stability index'):
    """
    Plots a chosen feature (e.g., precision) and stability index against the number of estimators for different n_feats values side by side,
    with the chosen feature on the left y-axis and the stability index on the right y-axis. Adjusts legend names based on the 'hpo' column
    and merges legends into one in the lower right corner.

    Parameters:
    - df: DataFrame containing the columns 'n_estimators', 'n_feats', 'precision', 'stability index', and 'hpo'.
    - ax: The matplotlib axes object where the plot will be drawn.
    - fontsize_title: Font size for the title.
    - fontsize_axes: Font size for the axes labels.
    - primary_feat: The primary feature to be plotted (e.g., 'precision').
    - secondary_feat: The secondary feature to be plotted on the second y-axis (e.g., 'stability index').
    """
    ax2 = ax.twinx()

    # Define styles and colors
    styles = {'Benchmark': '-', 'Our model': '-'}
    colors = {'Benchmark': '0.2', 'Our model': '0.7'}
    secondary_styles = {'Benchmark': '--', 'Our model': '--'}

    for (n_feats, hpo), group in df.groupby(['n_feats', 'hpo']):
        label = f"{hpo} (n_feats={n_feats})"
        ax.plot(group['n_estimators'], group[primary_feat], styles[hpo], color=colors[hpo], label=f"{primary_feat}, {label}")
        ax2.plot(group['n_estimators'], group[secondary_feat], secondary_styles[hpo], color=colors[hpo], alpha=0.8, label=f"{secondary_feat}, {label}")

    # Set axis labels and title
    ax.set_xlabel('Number of Estimators', fontsize=fontsize_axes)
    ax.set_ylabel(primary_feat.capitalize(), fontsize=fontsize_axes)
    ax2.set_ylabel(secondary_feat.capitalize(), fontsize=fontsize_axes)
    ax.set_title(df['dataset'].iloc[0].capitalize(), fontsize=fontsize_title)

    # Set tick parameters
    ax.tick_params(axis='x', labelsize=fontsize_axes)
    ax.tick_params(axis='y', labelsize=fontsize_axes)
    ax2.tick_params(axis='y', labelsize=fontsize_axes)

    # Collect and merge legends from both axes
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles + handles2, labels + labels2, fontsize=fontsize_axes - 2, loc='lower right')

    # Set y-ticks for both y-axes starting from 0
    min_primary, max_primary = df[primary_feat].min(), df[primary_feat].max()
    # y_ticks = [i * 0.1 for i in range(max(0, int(min_primary * 10) - 1), min(1, int(max_primary * 10) + 1))]
    y_ticks = [i * 0.1 for i in range(0, 7)]
    #ax.set_yticks(y_ticks)
    #ax2.set_yticks(y_ticks)
