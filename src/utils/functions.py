import numpy as np
import pandas as pd


def generate_estimators_max_features(total_features, steps, min_value=2):
    """
    Generate a list of feature estimator counts based on percentage steps.

    Parameters:
    - total_features: int, total number of features.
    - steps: int, how many steps (percentages) to include up to 100%.
    - min_value: int, minimum value for any step, default is 2.

    Returns:
    - List of integers representing feature estimator counts at various percentages of the total,
      with a minimum value enforced.
    """
    percentages = [i / steps for i in range(1, steps + 1)]  # Generate percentages based on the number of steps
    estimators = [max(min_value, int(total_features * pct)) for pct in percentages]  # Calculate estimators and enforce minimum value
    return estimators


def add_custom_repeating_sequence(df, new_column_name, sequence_length):
    """
    Adds a new column with a repeating sequence to the DataFrame.
    """
    sequence = np.tile(np.arange(1, sequence_length + 1), len(df) // sequence_length + 1)[:len(df)]
    df[new_column_name] = sequence
    return df


def add_sequence_to_dataframe(df, sequence, column_name):
    """
    Adds a custom sequence to the DataFrame, repeating it to match the DataFrame's length.
    """
    full_sequence = np.resize(sequence, len(df))
    df[column_name] = full_sequence
    return df


def prepare_subsets(df, ranks, max_feats):
    """
    Prepares and returns subsets of the DataFrame based on specified ranks and max_feats.
    """
    subsets = {}
    for rank in ranks:
        for max_feat in max_feats:
            key = f"df{rank}_{max_feat}"
            subset = df[(df['rank_feats'] == rank) & (df['rank_max_feats'] == max_feat)].copy()
            subset = subset.explode('shap_stab')
            subset['shap_stab'] = pd.to_numeric(subset['shap_stab'])
            subsets[key] = subset
    return subsets