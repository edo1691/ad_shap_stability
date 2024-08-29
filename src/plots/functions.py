import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plots_metrics(metrics_df, feat_imp=None,
                  metrics_list=['f1-score', 'recall', 'precision', 'roc_auc', 'smli', 'smli_all'],
                  x_axis_line_plots=None, x_axis_boxplots=None):
    if feat_imp is not None:
        selected_feats = []
        for imp in feat_imp:
            closest_feat = \
            metrics_df.iloc[(metrics_df['n_features_cum_shap_percentage'] - imp).abs().argsort()[:1]]['n_feat'].values[
                0]
            selected_feats.append(closest_feat)
        unique_features = sorted(set(selected_feats), reverse=True)
    else:
        unique_features = metrics_df['n_feat'].unique()

    # Set default x-axes if not provided
    if x_axis_line_plots is None:
        x_axis_line_plots = sorted(metrics_df['n_estimators'].unique())
    if x_axis_boxplots is None:
        x_axis_boxplots = sorted(metrics_df['n_estimators'].unique())

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

    # Create a blue color palette
    blue_palette = sns.color_palette("Blues", n_colors=len(unique_features))

    # Plot the non-smli_all metrics
    for m, metric in enumerate(non_smli_all_metrics):
        for i, n_feat in enumerate(unique_features):
            ax = axes[m, i] if total_rows > 1 else axes[i]
            subset_df = metrics_df[metrics_df['n_feat'] == n_feat]

            # Plotting a line with points connected for metrics
            ax.plot(subset_df['n_estimators'], subset_df[metric], marker='o', linestyle='-', color=blue_palette[i],
                    label=f'{n_feat} Features')

            if m == 0:
                ax.set_title(f'{n_feat} Features')
            if i == 0:
                ax.set_ylabel(metric.replace('_', ' ').capitalize())
            ax.set_xlabel('Number of Estimators' if m == len(non_smli_all_metrics) - 1 else '')
            ax.set_xticks(x_axis_line_plots)
            ax.set_xlim([min(x_axis_line_plots), max(x_axis_line_plots)])
            ax.grid(True)

    # Add the stability boxplots in the last row only if smli_all is in metrics_list
    if include_smli_all:
        stability_row = total_rows - 1
        for i, n_feat in enumerate(unique_features):
            ax = axes[stability_row, i] if len(unique_features) > 1 or total_rows > 1 else axes[0]
            subset_df = metrics_df[metrics_df['n_feat'] == n_feat]

            # Prepare data for boxplot
            plot_data = []
            for n_estimators in x_axis_boxplots:
                # Extract smli_all for the given n_estimators
                smli_values = subset_df[subset_df['n_estimators'] == n_estimators]['smli_all'].values
                if len(smli_values) > 0:
                    plot_data.append(smli_values[0])  # Add the smli_all array directly
                else:
                    plot_data.append([])  # Empty list if no values available

            # Create the boxplot using the prepared data
            sns.boxplot(data=plot_data, ax=ax, palette=blue_palette)
            ax.set_title(f'{n_feat} Features' if stability_row == 0 else "")
            ax.set_xlabel('Number of Estimators')
            ax.set_xticks(range(len(x_axis_boxplots)))
            ax.set_xticklabels(x_axis_boxplots)
            ax.set_xlim([-0.5, len(x_axis_boxplots) - 0.5])  # Set x limits to fit all boxes
            ax.set_ylabel('Stability' if i == 0 else "")
            ax.grid(True)

    plt.tight_layout()
    plt.show()