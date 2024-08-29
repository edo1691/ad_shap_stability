import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plots_metrics(metrics_df, feat_imp=None,
                  metrics_list=None,
                  metrics_merge_list=None,
                  x_axis_line_plots=None, x_axis_boxplots=None):
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

    # Handle metrics_merge_list
    if metrics_merge_list is not None:
        merged_metrics = [m for m in metrics_list if m in metrics_merge_list]
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
                             figsize=(sum(plot_widths) * 5, 5 * total_rows),
                             gridspec_kw={'width_ratios': boxplot_widths})

    # Handle cases where there's only one subplot
    if total_rows == 1 and len(unique_features) == 1:
        axes = np.array([axes])
    elif total_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif len(unique_features) == 1:
        axes = np.expand_dims(axes, -1)

    # Create a blue color palette large enough to cover all metrics
    num_colors_needed = max(len(unique_features), len(merged_metrics))
    blue_palette = sns.color_palette("Blues", n_colors=num_colors_needed)

    # Extend the palette with grey if needed
    if len(merged_metrics) > num_colors_needed:
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
            else:
                row_min = min(row_min, subset_df[metric].min())
                row_max = max(row_max, subset_df[metric].max())
        # Add 10% margin to min and max
        margin = 0.1 * (row_max - row_min)
        y_limits.append((row_min - margin, row_max + margin))

    # Plot the metrics
    for m, metric in enumerate(non_smli_all_metrics):
        for i, n_feat in enumerate(unique_features):
            ax = axes[m, i] if total_rows > 1 else axes[i]
            subset_df = metrics_df[metrics_df['n_feat'] == n_feat]

            if metric == 'merged_metrics':
                for j, merged_metric in enumerate(merged_metrics):
                    ax.plot(subset_df['n_estimators'], subset_df[merged_metric], marker='o', linestyle='-', color=blue_palette[j],
                            label=merged_metric)
                ax.legend()
            else:
                ax.plot(subset_df['n_estimators'], subset_df[metric], marker='o', linestyle='-', color=blue_palette[i],
                        label=f'{n_feat} Features')

            if m == 0:
                ax.set_title(f'{n_feat} Features')
            if i == 0:
                ax.set_ylabel(metric.replace('_', ' ').capitalize() if metric != 'merged_metrics' else 'Metrics')
            if not include_smli_all and m == len(non_smli_all_metrics) - 1:
                ax.set_xlabel('Number of Estimators')
            else:
                ax.set_xlabel('' if include_smli_all else 'Number of Estimators')
            ax.set_xticks(x_axis_line_plots)
            ax.set_xlim([min(x_axis_line_plots), max(x_axis_line_plots)])
            ax.set_ylim(y_limits[m])  # Set the y-limits for all plots in this row with margin
            ax.grid(True)

    # Add the stability boxplots in the last row only if smli_all is in metrics_list
    if include_smli_all:
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
                        flattened_values = smli_values[0].astype(float)
                        row_min = min(row_min, np.min(flattened_values))
                        row_max = max(row_max, np.max(flattened_values))
                    else:
                        row_min = min(row_min, float(smli_values[0]))
                        row_max = max(row_max, float(smli_values[0]))

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
                        plot_data.append(smli_values[0].astype(float))
                    else:
                        plot_data.append([float(smli_values[0])])
                else:
                    plot_data.append([])

            sns.boxplot(data=plot_data, ax=ax, palette=blue_palette)
            ax.set_title(f'{n_feat} Features' if stability_row == 0 else "")
            ax.set_xlabel('Number of Estimators')
            ax.set_xticks(range(len(x_axis_boxplots)))
            ax.set_xticklabels(x_axis_boxplots)
            ax.set_xlim([-0.5, len(x_axis_boxplots) - 0.5])  # Set x limits to fit all boxes
            ax.set_ylim(boxplot_ylim)  # Apply the same y-limits to all boxplots in this row
            ax.set_ylabel('Stability' if i == 0 else "")
            ax.grid(True)

    plt.tight_layout()
    plt.show()
