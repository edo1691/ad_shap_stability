import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import shap  # Assuming a function `shap_ranks` exists or is defined elsewhere that uses SHAP for feature ranking


def fs_iforest_with_shap(df, contamination_percentage=None, excluded_cols=None, n_trees=100, max_samples=256,
                         n_iter_fs=5):
    """
    Performs feature selection using an Isolation Forest model and SHAP values to rank features based on their
    importance in predicting anomalies. It iteratively evaluates the importance of features using SHAP values and
    calculates F1 scores to rank features.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the dataset to be used, including target variable 'y'.
    contamination_percentage : list of float, optional
        A list of contamination percentages to use in the Isolation Forest model. Defaults to [0.5, 1, 1.5] if None.
    excluded_cols : list of str, optional
        A list of column names to exclude from the feature set. Defaults to None.
    n_trees : int, optional
        The number of trees to use in the Isolation Forest model. Defaults to 100.
    max_samples : int, optional
        The number of samples to draw from X to train each base estimator. Defaults to 256.
    n_iter_fs : int, optional
        The number of iterations for feature selection. Defaults to 5.

    Returns
    -------
    tuple
        - sorted_idx (numpy.ndarray): Indices of features sorted by their importance.
        - fi_shap (numpy.ndarray): SHAP values indicating the importance of each feature.
        - avg_f1_ranking (numpy.ndarray): Average F1 scores for the features across iterations.
    """
    # Set default values for optional parameters
    if contamination_percentage is None:
        contamination_percentage = [0.5, 1, 1.5]
    if excluded_cols is None:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores']
    else:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores'] + excluded_cols

    if 'y' in df.columns:
        # Prepare the feature matrix X and target vector y
        X = np.array(df.loc[:, ~df.columns.isin(excluded_cols_all)])
        y = np.array(df['y'])

    else:
        # Prepare the feature matrix X and target vector y
        df['y'] = 1
        X = np.array(df.loc[:, ~df.columns.isin(excluded_cols_all)])
        y = np.array(df['y'])

    feature_names = df.loc[:, ~df.columns.isin(excluded_cols_all)].columns.to_numpy()

    # Assuming `shap_ranks` is a custom function that uses SHAP values to rank features
    # This function needs to be defined elsewhere in the code
    sorted_idx, fi_shap, avg_f1_ranking = shap_ranks(X, y, feature_names, n_trees=n_trees, max_samples=max_samples,
                                                     n_iter=n_iter_fs)

    return sorted_idx, fi_shap, avg_f1_ranking


def shap_ranks(x, y, feat_names, n_trees=100, max_samples=256, n_iter=5):
    """
    Ranks features based on SHAP values from an Isolation Forest model and evaluates the model's performance.

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Target vector.
    feat_names : numpy.ndarray or list
        Names of the features corresponding to the columns in X.
    n_trees : int, optional
        The number of trees for the Isolation Forest model.
    max_samples : int, optional
        The number of samples to draw from X to train each base estimator of the Isolation Forest model.
    n_iter : int, optional
        Number of iterations for SHAP value computation and feature ranking.

    Returns
    -------
    sorted_idx : list
        List of feature names sorted by their importance.
    fi_shap_all : pandas.DataFrame
        DataFrame containing features and their average SHAP values, percentage values, cumulative values, and cumulative percentage values.
    avg_f1 : float
        Average F1 score across iterations.
    """
    f1_all = []
    fi_shap_all_frames = []

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    for _ in range(n_iter):
        # Initialize and fit the Isolation Forest model
        iforest = IsolationForest(n_estimators=n_trees, max_samples=max_samples, contamination='auto',
                                  random_state=np.random.RandomState())
        iforest.fit(X_train)

        # Explainer initialization (assumes tree-based model for SHAP)
        explainer = shap.TreeExplainer(iforest, X_train, model_output='raw')
        shap_values = explainer.shap_values(X_test)

        # Calculate mean absolute SHAP values for each feature
        shap_summary = np.abs(shap_values).mean(axis=0)
        fi_shap = pd.DataFrame(list(zip(feat_names, shap_summary)), columns=['feature', 'value'])
        fi_shap_all_frames.append(fi_shap)

        # Predictions and F1 score calculation
        y_pred = iforest.predict(X_test)
        y_pred = np.where(y_pred == 1, 0, 1)  # Adjusting labels for F1 score calculation
        f1_all.append(f1_score(y_test, y_pred))

    # Combine all SHAP value DataFrames and compute average values
    fi_shap_all = pd.concat(fi_shap_all_frames).groupby('feature')['value'].mean().reset_index()
    fi_shap_all = fi_shap_all.sort_values(by='value', ascending=False).reset_index(drop=True)

    # Calculate additional statistics
    fi_shap_all['per_value'] = (fi_shap_all['value'] / fi_shap_all['value'].sum()) * 100
    fi_shap_all['cum_value'] = fi_shap_all['value'].cumsum()
    fi_shap_all['cum_value_percentage'] = fi_shap_all['cum_value'] / fi_shap_all['value'].sum() * 100

    # Sorted list of feature names by importance
    sorted_idx = fi_shap_all['feature'].tolist()

    # Average F1 score across iterations
    avg_f1 = np.mean(f1_all)

    return sorted_idx, fi_shap_all, avg_f1


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
