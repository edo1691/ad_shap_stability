import pandas as pd
from sklearn.ensemble import IsolationForest
from src.metric.functions import metrics_iforest


def train_and_evaluate_iforest(train_data, dataset_id, hyper=None, fi_df=None, n_tree_estimators=None,
                               contamination_percentage=None, excluded_cols=None,
                               n_iter_fs=5, n_iter=1):
    """
    Trains and evaluates an Isolation Forest model on given training data with specified parameters,
    and calculates various metrics including AUC-ROC, F1-score, recall, precision, and confusion matrix.

    Parameters
    ----------
    :param train_data : DataFrame
        The training data including features and target variable 'y'.
    :param dataset_id : str
        Identifier for the dataset used to tailor hyperparameters.
    :param hyper : str
    :param fi_df : DataFrame, optional
        DataFrame containing selected features based on feature importance. If not provided, all features are used.
    :param n_tree_estimators : list, optional
        List of numbers of trees to use in the Isolation Forest model. Defaults to [100] if None.
    :param contamination_percentage : list, optional
        List of contamination percentages to use in the Isolation Forest model. Defaults to [1] if None.
    :param excluded_cols : list, optional
        List of column names to exclude from the training data. Defaults to None.
    :param n_iter_fs : int, optional
        Number of iterations for feature selection. Defaults to 5.
    :param n_iter : int, optional
        Number of iterations for training and evaluation. Defaults to 1.

    Returns
    -------
    """

    # Set default values for optional parameters
    if contamination_percentage is None:
        contamination_percentage = [1]
    if n_tree_estimators is None:
        n_tree_estimators = [100]
    if excluded_cols is None:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores']
    else:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores'] + excluded_cols

    # Determine the list of features to use based on feature importance DataFrame or all features if not provided
    if fi_df is None:
        feat_list = [list(train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)].columns)]
    else:
        feat_list = list(fi_df.feat_selected)

    # Prepare the feature matrix X and target vector y
    X = train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)]
    y = train_data['y']

    # Convert dataset ID to lowercase for consistent handling
    dataset_id = dataset_id.lower()

    # Initialize lists to store the results of each iteration
    n_tree_list, n_cont_list, n_iter_list, n_iter_fs_list, n_feats_list, n_roc_auc, iforest_stab_unif_median_list, \
    shap_iforest_stab_unif_median_list, auc_precision_recall_median_list, f1_median_list, recall_median_list, \
    precision_median_list, conf_m_list = ([] for _ in range(13))

    # Get hyperparameters specific to the dataset
    factor = hyper['contamination']
    # Adjust contamination percentages based on the dataset-specific factor
    contamination_percentage = [round(x * factor, 3) for x in contamination_percentage]

    # Iterate over the number of trees parameter
    for tree_number in n_tree_estimators:
        hyper['n_estimators'] = tree_number
        print(f'Iteration by tree number: {tree_number}')

        # Iterate over contamination percentages
        for var_contamination in contamination_percentage:
            hyper['contamination'] = var_contamination
            print(f'  Iteration by contamination: {var_contamination}')

            # Iterate over selected feature sets
            for feat_selected in feat_list:
                print('    Number of featured:', len(feat_selected))
                train_data = X[feat_selected]
                train_data['y'] = y  # Add the target variable 'y' back to the training data

                # Perform model training and evaluation for each iteration
                for j in range(n_iter):
                    # Train the Isolation Forest model and make predictions
                    model, y_pred, y_scores, y_decision = train_and_predict_iforest(train_data, hyper,
                                                                                    excluded_cols_all,
                                                                                    random_state=j)

                    # Add predictions to the training data for evaluation
                    train_data['y_pred'] = y_pred
                    train_data['y_deci'] = y_decision
                    train_data['prediction'] = train_data.apply(def_outlier, axis=1)
                    train_data['y_scores'] = y_scores

                    # Calculate evaluation metrics
                    iforest_report, conf_m, roc_auc, iforest_stab_unif, shap_iforest_stab_unif = metrics_iforest(
                        train_data.loc[:, ~train_data.columns.isin(excluded_cols)], model, hyper)

                    # Store the results of this iteration
                    n_tree_list.append(tree_number)
                    n_cont_list.append(var_contamination)
                    n_iter_list.append(j + 1)
                    n_iter_fs_list.append(n_iter_fs)
                    n_feats_list.append(len(feat_selected))
                    n_roc_auc.append(roc_auc)
                    iforest_stab_unif_median_list.append(iforest_stab_unif)
                    shap_iforest_stab_unif_median_list.append(shap_iforest_stab_unif)
                    f1_median_list.append(iforest_report['1']['f1-score'])
                    recall_median_list.append(iforest_report['1']['recall'])
                    precision_median_list.append(iforest_report['1']['precision'])
                    conf_m_list.append(conf_m)

                # After all iterations, compile the results into a DataFrame
                result_df = pd.DataFrame({
                    'n_estimators': n_tree_list,
                    'contamination': n_cont_list,
                    'n_feats': n_feats_list,
                    'n_iter': n_iter_list,
                    'n_iter_fs': n_iter_fs_list,
                    'roc_auc': n_roc_auc,
                    'iforest_stab_unif_median': iforest_stab_unif_median_list,
                    'shap_iforest_stab_unif_median': shap_iforest_stab_unif_median_list,
                    'f1_median': f1_median_list,
                    'recall_median': recall_median_list,
                    'precision_median': precision_median_list,
                    'confusion_matrix': conf_m_list
                })

    # Return the compiled results DataFrame
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