import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import seaborn as sns


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


def boxplot_stability(df0, df1, ax, fontsize_title=8, fontsize_axes=8, color_without_fs='steelblue', color_with_fs='skyblue', title='', save_path=None):
    """
    Generates a 2x1 grid of plots for comparing stability metrics between dataframes without and with feature selection.

    Parameters:
    - df0: pandas.DataFrame without feature selection.
    - df1: pandas.DataFrame with feature selection.
    - ax: The Axes object or array of Axes objects on which to draw the plots.
    - fontsize: Font size for plot text elements. Defaults to 8.
    - color_without_fs: Color for the boxplot without feature selection. Defaults to 'steelblue'.
    - color_with_fs: Color for the boxplot with feature selection. Defaults to 'skyblue'.
    - title: Title for the entire figure. Optional.
    - save_path: Path where to save the figure. If None, the figure is not saved. Optional.
    """

    # Plot 1: SHAP Stability vs Number of Estimators (Without FS)
    sns.boxplot(data=df0, x='n_estimators', y='shap_stab', color=color_without_fs, ax=ax[0])
    ax[0].set_title('Without Feature Selection', fontsize=fontsize_title)
    ax[0].set_xlabel('Number of Estimators', fontsize=fontsize_axes)
    ax[0].set_ylabel('Stability', fontsize=fontsize_axes)
    ax[0].tick_params(axis='x', labelsize=fontsize_axes)
    ax[0].tick_params(axis='y', labelsize=fontsize_axes)

    # Plot 2: SHAP Stability vs Number of Estimators (With FS)
    sns.boxplot(data=df1, x='n_estimators', y='shap_stab', color=color_with_fs, ax=ax[1])
    ax[1].set_title('With Feature Selection', fontsize=fontsize_title)
    ax[1].set_xlabel('Number of Estimators', fontsize=fontsize_axes)
    ax[1].set_ylabel('Stability', fontsize=fontsize_axes)
    ax[1].tick_params(axis='x', labelsize=fontsize_axes)
    ax[1].tick_params(axis='y', labelsize=fontsize_axes)

    # Adjust layout
    plt.tight_layout()

    # Optionally save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plots
    plt.show()


def lineplot_stability(dataframes, ax, fontsize_title=12, fontsize_axes=8, colors=None, labels=None, save_path=None):
    """
    Generates line plots for comparing stability metrics across different configurations.

    Parameters:
    - dataframes: A list of pandas.DataFrame objects to plot. Each DataFrame represents a different scenario.
    - ax: Array of Axes objects on which to draw the plots. Should be of length 2 for this function.
    - fontsize: Font size for plot text elements. Defaults to 12.
    - colors: A list of colors for the plots. Length should match the number of dataframes.
    - labels: A list of labels for the legend, corresponding to each dataframe.
    - save_path: Path where to save the figure. If None, the figure is not saved. Optional.
    """
    # Set default colors and labels if none provided
    if colors is None:
        colors = ['steelblue', 'skyblue', 'green', 'orange', 'yellow', 'purple']
    if labels is None:
        labels = ['Without FS', 'With FS-100%', 'With FS-80%', 'With FS-60%', 'With FS-40%', 'With FS-20%']

    # Mean SHAP Stability vs Number of Estimators and Max features selected (AUPRC)
    for df, color, label in zip(dataframes, colors, labels):
        sns.pointplot(data=df, x='n_estimators', y='roc_auc', color=color, markers='', linestyles='--', ci=None,
                      ax=ax[0], label=label)
    ax[0].set_title('AUPRC vs Number of Estimators', fontsize=fontsize_title)
    ax[0].set_xlabel('Number of Estimators', fontsize=fontsize_axes)
    ax[0].set_ylabel('ROC', fontsize=fontsize_axes)
    ax[0].legend(title='Criteria', fontsize=fontsize_axes, loc='lower right', ncol=3)
    ax[0].tick_params(axis='x', labelsize=fontsize_axes)
    ax[0].tick_params(axis='y', labelsize=fontsize_axes)

    # Mean SHAP Stability vs Number of Estimators
    for df, color, label in zip(dataframes, colors, labels):
        sns.pointplot(data=df, x='n_estimators', y='stability index', color=color, markers='', linestyles='--', ci=None,
                      ax=ax[1], label=label)
    ax[1].set_title('Mean Stability vs Number of Estimators', fontsize=fontsize_title)
    ax[1].set_xlabel('Number of Estimators', fontsize=fontsize_axes)
    ax[1].set_ylabel('Mean Stability', fontsize=fontsize_axes)
    ax[1].legend(title='Criteria', fontsize=fontsize_axes, loc='lower right', ncol=3)
    ax[1].tick_params(axis='x', labelsize=fontsize_axes)
    ax[1].tick_params(axis='y', labelsize=fontsize_axes)

    # Adjust layout
    plt.tight_layout()

    # Optionally save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plots
    plt.show()
