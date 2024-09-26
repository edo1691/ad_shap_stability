import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import os


def plots_metrics(metrics_df, dataset_id, feat_imp=None, metrics_list=None, metrics_merge_list=None,
                  x_axis_line_plots=None, x_axis_boxplots=None, x_axis_start=None, x_axis_end=None,
                  save_plot=None, save_dir=None):  # New parameters added for saving plots

    if metrics_list is None:
        metrics_list = ['f1-score', 'recall', 'precision', 'roc_auc', 'smli', 'smli_all']
    if feat_imp is not None:
        selected_feats = []
        for imp in feat_imp:
            closest_feat = metrics_df.iloc[
                (metrics_df['n_features_cum_shap_percentage'] - imp).abs().argsort()[:1]
            ]['n_feat'].values[0]
            selected_feats.append(closest_feat)
        unique_features = sorted(set(selected_feats), reverse=True)
    else:
        unique_features = metrics_df['n_feat'].unique()

    # Ensure all relevant columns are numeric
    for col in metrics_list + ['n_feat', 'n_estimators']:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(pd.to_numeric, errors='coerce')

    # Set default x-axes if not provided
    if x_axis_line_plots is None:
        x_axis_line_plots = sorted(metrics_df['n_estimators'].unique())
    if x_axis_boxplots is None:
        x_axis_boxplots = sorted(metrics_df['n_estimators'].unique())

    # Calculate the minimum x-axis start from boxplots if not provided
    if x_axis_start is None:
        x_axis_start = min(x_axis_boxplots)

    # Calculate the maximum x-axis end from boxplots
    if x_axis_end is None:
        x_axis_end = max(x_axis_boxplots)

    # Handle metrics_merge_list
    if metrics_merge_list is not None:
        merged_metrics = [m for m in metrics_list if m in metrics_merge_list]
        if 'smli_all' in metrics_merge_list:
            metrics_list = [m for m in metrics_list if m not in metrics_merge_list or m == 'smli_all']
        else:
            metrics_list = [m for m in metrics_list if m not in metrics_merge_list]
        metrics_list.insert(0, 'merged_metrics')  # Insert a placeholder for merged metrics
    else:
        merged_metrics = []

    # Determine the number of rows: check if smli_all is in metrics_list
    include_smli_all = 'smli_all' in metrics_list
    non_smli_all_metrics = [m for m in metrics_list if m != 'smli_all']
    total_rows = len(non_smli_all_metrics) + (
        1 if include_smli_all else 0)  # Adding one row for stability boxplots if smli_all is present

    # Calculate the width of each subplot based on the number of n_estimators
    plot_widths = [len(x_axis_line_plots) / 10 for _ in unique_features]  # Adjust the divisor to scale plot width
    boxplot_widths = [len(x_axis_boxplots) / 10 for _ in unique_features]  # Adjust the divisor to scale plot width

    # Set up the figure with varying widths based on n_estimators
    fig, axes = plt.subplots(total_rows, len(unique_features),
                             figsize=(sum(plot_widths) * 12, 4 * total_rows),  # Increase the size here
                             gridspec_kw={'width_ratios': boxplot_widths})

    # Handle cases where there's only one subplot
    if total_rows == 1 and len(unique_features) == 1:
        axes = np.array([axes])
    elif total_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif len(unique_features) == 1:
        axes = np.expand_dims(axes, -1)

    # Create a blue color palette large enough to cover all metrics
    num_colors_needed = max(len(unique_features), len(merged_metrics) + (1 if 'smli_all' in metrics_merge_list else 0))
    blue_palette = sns.color_palette("Blues", n_colors=num_colors_needed)

    # Extend the palette with grey if needed
    if len(merged_metrics) + (1 if 'smli_all' in metrics_merge_list else 0) > num_colors_needed:
        blue_palette.extend(sns.color_palette("Greys", n_colors=len(merged_metrics) - num_colors_needed))

    # Calculate y-axis limits for each row with a 10% margin
    y_limits = []
    for m, metric in enumerate(non_smli_all_metrics):
        row_min = np.inf
        row_max = -np.inf
        for i, n_feat in enumerate(unique_features):
            subset_df = metrics_df[metrics_df['n_feat'] == n_feat]
            if metric == 'merged_metrics':
                for merged_metric in merged_metrics:
                    row_min = min(row_min, subset_df[merged_metric].min())
                    row_max = max(row_max, subset_df[merged_metric].max())
                if 'smli_all' in metrics_merge_list:
                    row_min = min(row_min, subset_df['smli_all'].min())
                    row_max = max(row_max, subset_df['smli_all'].max())
            else:
                row_min = min(row_min, subset_df[metric].min())
                row_max = max(row_max, subset_df[metric].max())
        # Add 10% margin to min and max
        margin = 0.1 * (row_max - row_min)
        y_limits.append((row_min - margin, row_max + margin))

    # Define a list of markers to use for different lines
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'h']  # Add more markers as needed

    # Plot the metrics
    for m, metric in enumerate(non_smli_all_metrics):
        for i, n_feat in enumerate(unique_features):
            ax = axes[m, i] if total_rows > 1 else axes[i]
            subset_df = metrics_df[metrics_df['n_feat'] == n_feat]

            if metric == 'merged_metrics':
                for j, merged_metric in enumerate(merged_metrics):
                    marker = markers[j % len(markers)]  # Cycle through the list of markers
                    ax.plot(subset_df['n_estimators'], subset_df[merged_metric], marker=marker, linestyle='-',
                            color=blue_palette[j],
                            label=merged_metric)
                if 'smli_all' in metrics_merge_list:
                    ax.plot(subset_df['n_estimators'], subset_df['smli_all'], marker='X', linestyle='-', color='red',
                            label='smli_all')
                ax.legend()
            else:
                marker = markers[i % len(markers)]  # Cycle through the list of markers
                ax.plot(subset_df['n_estimators'], subset_df[metric], marker=marker, linestyle='-', color=blue_palette[i],
                        label=f'{n_feat} Features')

            # Modify the title to include dataset_id and n_feat
            if m == 0:
                ax.set_title(f'{dataset_id} - {n_feat} Features')
            if i == 0:
                ax.set_ylabel(metric.replace('_', ' ').capitalize() if metric != 'merged_metrics' else 'Metrics')
            if not include_smli_all and m == len(non_smli_all_metrics) - 1:
                ax.set_xlabel('Number of Estimators')
            else:
                ax.set_xlabel('' if include_smli_all else 'Number of Estimators')

            # Set the same X-axis limits for all plots
            ax.set_xlim([x_axis_start, x_axis_end])
            ax.set_xticks(x_axis_line_plots)
            ax.set_ylim(y_limits[m])  # Set the y-limits for all plots in this row with margin
            ax.grid(True)

    # Add the stability boxplots in the last row only if smli_all is in metrics_list and not in metrics_merge_list
    if include_smli_all and 'smli_all' not in metrics_merge_list:
        stability_row = total_rows - 1
        row_min = np.inf
        row_max = -np.inf

        # Calculate y-axis limits for the entire row of boxplots
        for i, n_feat in enumerate(unique_features):
            subset_df = metrics_df[metrics_df['n_feat'] == n_feat]
            for n_estimators in x_axis_boxplots:
                smli_values = subset_df[subset_df['n_estimators'] == n_estimators]['smli_all'].values
                if len(smli_values) > 0:
                    if isinstance(smli_values[0], np.ndarray):
                        flattened_values = np.concatenate([v.astype(float).flatten() for v in smli_values])
                    else:
                        flattened_values = np.array([float(v) for v in smli_values])
                    row_min = min(row_min, np.min(flattened_values))
                    row_max = max(row_max, np.max(flattened_values))

        # Add 10% margin to the min and max
        margin = 0.1 * (row_max - row_min)
        boxplot_ylim = (row_min - margin, row_max + margin)

        # Create boxplots with consistent y-limits
        for i, n_feat in enumerate(unique_features):
            ax = axes[stability_row, i] if len(unique_features) > 1 or total_rows > 1 else axes[0]
            subset_df = metrics_df[metrics_df['n_feat'] == n_feat]
            plot_data = []
            for n_estimators in x_axis_boxplots:
                smli_values = subset_df[subset_df['n_estimators'] == n_estimators]['smli_all'].values
                if len(smli_values) > 0:
                    if isinstance(smli_values[0], np.ndarray):
                        plot_data.append(np.concatenate([v.astype(float).flatten() for v in smli_values]))
                    else:
                        plot_data.append([float(v) for v in smli_values])
                else:
                    plot_data.append([])

            sns.boxplot(data=plot_data, ax=ax, palette=blue_palette)
            ax.set_title(f'{dataset_id} - {n_feat} Features' if stability_row == 0 else "")
            ax.set_xlabel('Number of Estimators')
            ax.set_xticks(range(len(x_axis_boxplots)))
            ax.set_xticklabels(x_axis_boxplots)
            ax.set_xlim([-0.5, len(x_axis_boxplots) - 0.5])  # Set x limits to fit all boxes
            ax.set_ylim(boxplot_ylim)  # Apply the same y-limits to all boxplots in this row
            ax.set_ylabel('Stability' if i == 0 else "")
            ax.grid(True)

    plt.tight_layout()

    # Save plot if save_plot is specified
    if save_plot:
        if save_dir is None:
            save_dir = "./"  # Default to current directory if save_dir is not provided
        save_path = os.path.join(save_dir, f"{dataset_id}_metrics_plot.png")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# Function to load dataset and plot metrics
def process_and_plot(dataset_id, data_root, feat_imp=None, metrics_list=None, metrics_merge_list=None,
                     x_axis_start=12.25, x_axis_end=212.25, save_plot=None, save_dir=None):  # New parameters added

    # Construct file paths
    path_fi_shap = os.path.join(data_root, "outputs", f"{dataset_id}_fi_shap")
    path_shap = os.path.join(data_root, "outputs", f"{dataset_id}_shap.parquet")

    # Load the feature importance and SHAP values
    features = pd.read_parquet(path_fi_shap)
    df = pd.read_parquet(path_shap)

    # Plot the metrics
    plots_metrics(
        metrics_df=df,
        dataset_id=dataset_id,
        feat_imp=feat_imp,  # Adjust this as needed
        metrics_list=metrics_list,
        metrics_merge_list=metrics_merge_list,
        x_axis_start=x_axis_start,
        x_axis_end=x_axis_end,
        save_plot=save_plot,  # Pass the new save_plot parameter
        save_dir=save_dir     # Pass the new save_dir parameter
    )
