import pandas as pd
from sklearn.ensemble import IsolationForest
from src.metric.functions import metrics_iforest
from src.utils.functions import generate_estimators_max_features
from typing import List, Optional, Dict, Any


def train_and_evaluate_iforest(train_data: pd.DataFrame, dataset_id: str, hyper: Optional[Dict[str, Any]] = None,
                               fi_df: Optional[pd.DataFrame] = None, n_feat_estimators: Optional[List[int]] = None,
                               n_tree_estimators: Optional[List[int]] = None,
                               contamination_percentage: Optional[List[float]] = None,
                               excluded_cols: Optional[List[str]] = None, n_iter_fs: int = 5,
                               n_iter: int = 1) -> pd.DataFrame:
    """
    Train and evaluate an Isolation Forest model on given training data and parameters.

    Args:
    - train_data: DataFrame containing the training data.
    - dataset_id: Unique identifier for the dataset.
    - hyper: Dictionary containing hyperparameters for the model.
    - fi_df: DataFrame containing feature importance scores.
    - n_feat_estimators: List of numbers of feature estimators to use.
    - n_tree_estimators: List of numbers of tree estimators to use.
    - contamination_percentage: List of contamination percentages to consider.
    - excluded_cols: List of column names to exclude from training.
    - n_iter_fs: Number of iterations for feature selection.
    - n_iter: Number of iterations for training and evaluation.

    Returns:
    - DataFrame containing the evaluation metrics for each iteration.
    """
    # Set default values
    contamination_percentage = contamination_percentage or [1]
    n_tree_estimators = n_tree_estimators or [100]
    n_feat_estimators = n_feat_estimators or 'auto'
    excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores'] + (excluded_cols or [])

    # Determine feature list
    temp_feat_list = [train_data.columns.difference(excluded_cols_all).tolist()] if fi_df is None else fi_df[
        'feat_selected'].tolist()

    seen = set()
    feat_list = []
    for item in temp_feat_list:
        # Convert each item (list or array) to a tuple for hashing
        item_tuple = tuple(item)
        if item_tuple not in seen:
            seen.add(item_tuple)
            # Add the original item (list or array) to the result
            feat_list.append(item)

    # Ensure 'y' column is present
    if 'y' not in train_data.columns:
        train_data['y'] = 1
    X = train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)]
    y = train_data['y']

    # Convert dataset ID to lowercase
    dataset_id = dataset_id.lower()
    results = []

    # Adjust contamination based on dataset-specific factor
    factor = hyper.get('contamination', 1)
    contamination_percentage = [round(x * factor, 3) for x in contamination_percentage]

    for tree_number in n_tree_estimators:
        hyper['n_estimators'] = tree_number
        print(f'Iteration by tree number: {tree_number}')
        for var_contamination in contamination_percentage:
            hyper['contamination'] = var_contamination
            print(f'  Iteration by contamination: {var_contamination}')
            for feat_selected in feat_list:
                print(f'    Number of featured: {len(feat_selected)}')
                train_subset = X[feat_selected]
                train_subset['y'] = y
                temp_list = generate_estimators_max_features(len(feat_selected), 5)
                n_feat_estimators = list(dict.fromkeys(temp_list))

                for feat_number in n_feat_estimators:
                    hyper['max_features'] = feat_number
                    print(f'     Iteration by feat number: {feat_number}')
                    for j in range(n_iter):
                        # Assuming train_and_predict_iforest and metrics_iforest are defined elsewhere
                        model, y_pred, y_scores, y_decision = train_and_predict_iforest(train_subset, hyper)

                        train_subset['y_pred'] = y_pred
                        train_subset['y_deci'] = y_decision
                        train_subset['prediction'] = train_subset.apply(def_outlier, axis=1)
                        train_subset['y_scores'] = y_scores

                        metrics = metrics_iforest(train_subset, model)
                        metrics.update({'n_estimators': tree_number, 'max_feats': feat_number,
                                        'contamination': var_contamination, 'n_feats': len(feat_selected),
                                        'n_iter': j + 1, 'n_iter_fs': n_iter_fs})
                        results.append(metrics)

    result_df = pd.DataFrame(results)
    return result_df


def train_and_predict_iforest(train_data, hyper, excluded_cols=None, random_state=12345):
    """
    Train an Isolation Forest model and make predictions on the provided dataset.

    Parameters:
    - train_data: DataFrame containing the training data.
    - hyper: Dictionary of hyperparameters for the Isolation Forest model.
    - excluded_cols_all: List of columns to exclude during training and prediction.
    - random_state: Random seed for reproducibility.

    Returns:
    - model: Trained Isolation Forest model.
    - y_pred: Predicted labels.
    - y_scores: Anomaly scores.
    """

    if excluded_cols is None:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores']
    else:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores'] + excluded_cols
    # Training
    model = IsolationForest(**hyper, random_state=random_state)
    model.fit(train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)])

    # Prediction
    y_pred = model.predict(train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)])
    y_scores = -model.score_samples(train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)])
    y_decision = -model.decision_function(train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)])

    return model, y_pred, y_scores, y_decision


def def_outlier(df):
    if (df['y_pred'] in [-1]):
        val = 1
    else:
        val = 0
    return val