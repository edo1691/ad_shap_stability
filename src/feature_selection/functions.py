import pandas as pd
import numpy as np
import shap


def shap_feature_selection(model, X_train, X_test, feature_names, agnostic=False):
    """
    Perform feature selection using SHAP values and return a DataFrame with selected features
    and their corresponding SHAP values, percentages, and cumulative values.

    Parameters:
    - model: Trained model (e.g., RandomForest, XGBoost).
    - X_train: Training dataset.
    - X_test: Testing dataset.
    - feature_names: List of feature names.
    - agnostic: Whether to use model-agnostic SHAP (e.g., KernelExplainer).

    Returns:
    - df: DataFrame with selected features, SHAP values, and calculated statistics.
    """

    # Calculate SHAP values
    if agnostic:
        explainer = shap.KernelExplainer(model.predict, X_train)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)  # Assuming binary classification; use [0] for negative class

    # Compute mean absolute SHAP values for each feature
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Create a DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'value': mean_shap_values
    })

    # Sorting by value in descending order
    df = df.sort_values(by='value', ascending=False).reset_index(drop=True)

    # Calculate per_value (percentage of each value in the total)
    total_value = df['value'].sum()
    df['per_value'] = df['value'] / total_value * 100

    # Calculate cumulative values and cumulative percentages
    df['cum_value'] = df['value'].cumsum()
    df['cum_value_percentage'] = df['per_value'].cumsum()

    return df


def process_fi(df, fs=None):
    """
    Processes a DataFrame of feature importances to identify and flag features at specified intervals based on their
    cumulative percentage contribution. It then filters to these features, calculates additional metrics,
    selects features based on their ranks, and duplicates the last row with modifications.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing feature importance metrics including 'per_value' and 'cum_value_percentage' columns.
    fs : float, optional
        Step size for flagging features based on cumulative percentage intervals. If None, the minimum positive
        'per_value' in the DataFrame is used.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with selected features and modified last row.
    """
    # Copy dataframe for manipulation and for retaining original feature names
    df_fi = df.copy()
    df_fi_names = df.copy()

    # Determine the step size if not provided or set to 0
    if fs is None or fs == 0:
        fs = df_fi[df_fi.per_value > 0].per_value.min()

    # Create a new column 'flag' initialized with NaN
    df_fi['flag'] = np.nan

    # Calculate multiples of fs from fs to 105 (assuming 100% is the maximum)
    multiples = np.arange(fs, 105, fs)

    # Find the nearest index for each multiple of fs and flag them
    for multiple in multiples:
        nearest_index = df_fi['cum_value_percentage'].sub(multiple).abs().idxmin()
        df_fi.at[nearest_index, 'flag'] = 1

    # Filter the DataFrame to only include flagged features and reset index
    df_fi = df_fi[df_fi['flag'] == 1].reset_index()
    df_fi = df_fi.rename(columns={'index': 'n_feats'})
    df_fi['n_feats'] += 1

    # Calculate percentage of features based on 'n_feats'
    df_fi['n_feats_percentage'] = (df_fi['n_feats'] / df_fi['n_feats'].max()) * 100

    # Extract the first X elements from 'feature' column based on 'n_feats'
    df_fi['feat_selected'] = df_fi.apply(lambda row: list(df_fi_names['feature'][:row['n_feats']]), axis=1)

    # Clean the DataFrame by dropping unnecessary columns
    df_fi = df_fi.drop(columns=['feature', 'value', 'per_value', 'flag'], axis=1)

    # Duplicate the last row and modify 'n_feats' and 'feat_selected' for the new row
    last_row_modified = df_fi.iloc[-1].copy()
    last_row_modified['n_feats'] = len(df_fi_names)
    last_row_modified['feat_selected'] = list(df_fi_names['feature'])
    df_fi = df_fi.append(last_row_modified, ignore_index=True)

    return df_fi
