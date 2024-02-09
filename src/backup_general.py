import os
import logging

import pandas as pd
import numpy as np
import os
import sys
from os.path import join
import json

import shap
from shap_selection import feature_selection
import sage


from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn import metrics
from sklearn.model_selection import train_test_split

from src.stability.functions import stability_measure #, stability_measure_shap
from src.stability.functions_shap import stability_measure_shap
import src.model_interpretability.interpretability_module as interp
from src.model_interpretability.utils import diffi_ranks, get_fs_dataset, fs_datasets_hyperparams


logger = logging.getLogger(__name__)


def def_outlier(df):
    if (df['y_pred'] in [-1]):
        val = 1
    else:
        val = 0
    return val


def metrics_iforest(data, model, hyper, stratify=True):
    excluded = ['y', 'y_pred', 'prediction', 'y_scores']
    # Iforest report: precision, recall and f1_score
    iforest_report = classification_report(data['y'], data['prediction'], target_names=['0', '1'], output_dict=True)

    # Confusion matrix
    # conf_m = confusion_matrix(data['y'], data['prediction'])
    conf_m = 1

    fpr, tpr, thresholds = metrics.roc_curve(data['y'], data['y_deci'])
    roc_auc = metrics.auc(fpr, tpr)

    # Stability Index
    # Split the dataset into training and testing sets while stratifying based on the target variable
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(data.drop(excluded, axis=1), data['y'], test_size=0.2, random_state=42, stratify=data['y'])
    else:
        X_train, X_test = train_test_split(data.drop(excluded, axis=1), test_size=0.33, random_state=42)

    stab_model, _ = stability_measure_model(X_train, X_test, model,
                                             gamma=hyper['contamination'],
                                             unif=True,
                                             iterations=5)

    stab_shap, _ = stability_measure_shap(X_train, X_test, model,
                                             gamma=hyper['contamination'],
                                             unif=True,
                                             iterations=5)

    return iforest_report, conf_m, roc_auc, stab_model, stab_shap


def train_and_predict_isolation_forest(train_data, hyper, excluded_cols=None, random_state=12345):
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


def train_and_evaluate_iforest(train_data, dataset_id, fi_df=None, n_tree_estimators=None,
                               contamination_percentage=None, excluded_cols=None,
                               n_iter_fs=5, n_iter=1):

    if contamination_percentage is None:
        contamination_percentage = [1]

    if n_tree_estimators is None:
        n_tree_estimators = [100]

    if excluded_cols is None:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores']
    else:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores'] + excluded_cols

    if fi_df is None:
        feat_list = list(train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)].columns)
        feat_list = [feat_list]
    else:
        feat_list = list(fi_df.feat_selected)

    X = train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)]
    y = train_data['y']

    dataset_id = dataset_id.lower()

    # Initialize lists to store metrics for each iteration
    n_tree_list, n_cont_list, n_iter_list, n_iter_fs_list, n_feats_list, n_roc_auc, iforest_stab_unif_median_list, shap_iforest_stab_unif_median_list, auc_precision_recall_median_list, f1_median_list, recall_median_list, precision_median_list, conf_m_list = [], [], [], [], [], [], [], [], [], [], [], [], []

    hyper = fs_datasets_hyperparams(dataset_id)
    factor = hyper['contamination']
    contamination_percentage = [round(x * factor, 3) for x in contamination_percentage]

    for tree_number in n_tree_estimators:
        hyper['n_estimators'] = tree_number
        print(f'Iteration by tree number: {tree_number}')

        for var_contamination in contamination_percentage:
            hyper['contamination'] = var_contamination
            print(f'  Iteration by contamination: {var_contamination}')

            for feat_selected in feat_list:
                print('    Number of featured:', len(feat_selected))
                train_data = X[feat_selected]
                train_data['y'] = y

                for j in range(n_iter):
                    # Training
                    model, y_pred, y_scores, y_decision = train_and_predict_isolation_forest(train_data, hyper, excluded_cols_all,
                                                                                             random_state=j)

                    # Add prediction to the dataframe
                    train_data['y_pred'] = y_pred
                    train_data['y_deci'] = y_decision
                    train_data['prediction'] = train_data.apply(def_outlier, axis=1)
                    train_data['y_scores'] = y_scores

                    # Calculate Metrics
                    iforest_report, conf_m, roc_auc, iforest_stab_unif, shap_iforest_stab_unif = metrics_iforest(
                        train_data.loc[:, ~train_data.columns.isin(excluded_cols)], model, hyper)

                    # Save metrics for each iteration
                    n_tree_list.append(tree_number)
                    n_cont_list.append(var_contamination)
                    n_iter_list.append(j + 1)
                    n_iter_fs_list.append(n_iter_fs)
                    n_feats_list.append(train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)].shape[1])
                    n_roc_auc.append(roc_auc)
                    iforest_stab_unif_median_list.append(iforest_stab_unif)
                    shap_iforest_stab_unif_median_list.append(shap_iforest_stab_unif)
                    f1_median_list.append(iforest_report['1']['f1-score'])
                    recall_median_list.append(iforest_report['1']['recall'])
                    precision_median_list.append(iforest_report['1']['precision'])
                    conf_m_list.append(conf_m)

                # Create DataFrame from lists
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

            # Create DataFrame
            result_df = pd.DataFrame(result_df)

    return result_df


def fs_iforest_with_shap(train_data, contamination_percentage=None, excluded_cols=None, n_trees=100, max_samples=256,
                         n_iter_fs=5, n_iter=1):
    if contamination_percentage is None:
        contamination_percentage = [0.5, 1, 1.5]

    if excluded_cols is None:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores']
    else:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores'] + excluded_cols

    X = np.array(train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)])
    y = np.array(train_data['y'])
    feature_names = np.array(train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)].columns)

    sorted_idx, fi_shap, avg_f1_ranking = shap_ranks(X, y, feature_names, n_trees=n_trees, max_samples=max_samples,
                                                     n_iter=n_iter_fs)

    return sorted_idx, fi_shap, avg_f1_ranking


def shap_ranks(X, y, feat_names, n_trees, max_samples, n_iter):
    f1_all, fi_shap_all, feature_names, feature_values = [], [], [], []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    for k in range(n_iter):
        # ISOLATION FOREST
        # fit the model
        iforest = IsolationForest(n_estimators=n_trees, max_samples=max_samples,
                                  contamination='auto', random_state=k)
        iforest.fit(X_train)

        # get SHAP rank
        feature_order = feature_selection.shap_select(iforest, X_train, X_test, feat_names, task='classification',
                                                      agnostic=False, background_size=0.1)

        feature_names = list(feature_order[0])
        feature_values = list(feature_order[1])

        fi_shap = pd.DataFrame(feature_values, index=feature_names).reset_index()
        fi_shap.columns = ['feature', 'value']
        fi_shap_all.append(fi_shap)

        # get predictions
        y_pred = np.array(iforest.decision_function(X) < 0).astype('int')
        # get performance metrics
        f1_all.append(f1_score(y, y_pred))

    # concatenate DataFrames along the rows
    fi_shap_all = pd.concat(fi_shap_all, ignore_index=True)

    # compute avg F1
    avg_f1 = np.mean(f1_all)

    # compute the scores
    fi_shap_all = fi_shap_all.groupby('feature')['value'].mean().reset_index()
    fi_shap_all = fi_shap_all.sort_values(by='value', ascending=False)
    fi_shap_all = fi_shap_all.reset_index(drop=True)

    sorted_idx = fi_shap_all['feature'].tolist()
    fi_shap_all['per_value'] = (fi_shap_all['value'] / fi_shap_all['value'].sum()) * 100
    fi_shap_all['cum_value'] = fi_shap_all['value'].cumsum()
    fi_shap_all['cum_value_percentage'] = (fi_shap_all['value'] / fi_shap_all['value'].sum()).cumsum() * 100

    return sorted_idx, fi_shap_all, avg_f1


def process_fi(df, fs=None):
    # Copy dataframe:
    df_fi = df.copy()
    df_fi_names = df.copy()

    if fs is None:
        fs = df_fi[df_fi.per_value > 0].per_value.min()

    if fs is 0:
        fs = df_fi[df_fi.per_value > 0].per_value.min()

    # Create a new column 'Flag' with 0 or 1
    # Find the nearest index for each multiple of 5
    multiples = np.arange(fs, 105, fs)
    nearest_indices = []

    for multiple in multiples:
        nearest_index = df_fi['cum_value_percentage'].sub(multiple).abs().idxmin()
        nearest_indices.append(nearest_index)

    # Mark the rows corresponding to the nearest indices in the 'flag' column
    df_fi.loc[nearest_indices, 'flag'] = 1

    df_fi = df_fi.reset_index()
    df_fi = df_fi.rename(columns={'index': 'n_feats'})
    df_fi['n_feats'] += 1
    df_fi = df_fi[df_fi.flag == 1]

    # Assuming 'cum_shap_value_percentage' and 'n_feats' columns exist in df_fi
    df_fi['n_feats_percentage'] = (df_fi['n_feats'] / df_fi['n_feats'].max()) * 100

    # Extract the first X elements from 'name_var' based on 'n_feats'
    df_fi['feat_selected'] = df_fi.apply(lambda row: list(df_fi_names['feature'][:row['n_feats']]), axis=1)

    df_fi = df_fi.drop(columns=['feature', 'value', 'per_value', 'flag'], axis=1)

    return df_fi


def fs_iforest_with_diffi(train_data, contamination_percentage=None, excluded_cols=None, n_trees=100, max_samples=256,
                          n_iter_fs=5, n_iter=1):
    if contamination_percentage is None:
        contamination_percentage = [0.5, 1, 1.5]
    if excluded_cols is None:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores']
    else:
        excluded_cols_all = ['y', 'y_pred', 'prediction', 'y_scores'] + excluded_cols

    X = np.array(train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)])
    y = np.array(train_data['y'])

    sorted_idx, scores, avg_f1_ranking = diffi_ranks(X, y, n_trees=n_trees, max_samples=max_samples, n_iter=n_iter_fs)

    sorted_df = train_data.loc[:, ~train_data.columns.isin(excluded_cols_all)]

    sorted_idx = list(np.array(sorted_df.iloc[:, sorted_idx].columns))
    # name_var = list(sorted_df.columns)
    fi_diffi = list(zip(sorted_idx, scores))

    fi_diffi = pd.DataFrame(fi_diffi, columns=['feature', 'value'])
    fi_diffi = fi_diffi.sort_values('value', ascending=False)
    fi_diffi = fi_diffi.reset_index(drop=True)
    fi_diffi['per_value'] = (fi_diffi['value'] / fi_diffi['value'].sum()) * 100
    fi_diffi['cum_value'] = fi_diffi['value'].cumsum()
    fi_diffi['cum_value_percentage'] = (fi_diffi['value'] / fi_diffi['value'].sum()).cumsum() * 100

    #sorted_idx = list(np.array(sorted_df.iloc[:, sorted_idx].columns))

    return sorted_idx, fi_diffi, avg_f1_ranking


def process_fi_diffi(fi_diffi_all):
    # Copy dataframe:
    fi_diffi_names = fi_diffi_all.copy()

    # Create a new column 'Flag' with 0 or 1
    # Find the nearest index for each multiple of 5
    multiples_of_5 = np.arange(5, 105, 5)
    nearest_indices = []

    for multiple in multiples_of_5:
        nearest_index = fi_diffi_all['cum_diffi_value_percentage'].sub(multiple).abs().idxmin()
        nearest_indices.append(nearest_index)

    # Mark the rows corresponding to the nearest indices in the 'flag' column
    fi_diffi_all.loc[nearest_indices, 'flag'] = 1

    fi_diffi_all = fi_diffi_all.reset_index()
    fi_diffi_all = fi_diffi_all.rename(columns={'index': 'n_feats'})
    fi_diffi_all['n_feats'] += 1
    fi_diffi_all = fi_diffi_all[fi_diffi_all.flag == 1]

    # Assuming 'cum_shap_value_percentage' and 'n_feats' columns exist in fi_shap_all
    fi_diffi_all['n_feats_percentage'] = (fi_diffi_all['n_feats'] / fi_diffi_all['n_feats'].max()) * 100
    fi_diffi_all['n_feats_diff'] = fi_diffi_all['n_feats'].diff().fillna(min(fi_diffi_all.n_feats))
    fi_diffi_all['cum_diffi_value_diff'] = fi_diffi_all['cum_diffi_value'].diff().fillna(
        min(fi_diffi_all.cum_diffi_value))
    fi_diffi_all['marginal_diffi_value'] = fi_diffi_all['cum_diffi_value_diff'] / fi_diffi_all['n_feats_diff']
    fi_diffi_all['marginal_diffi_value_percentage'] = (
                fi_diffi_all['marginal_diffi_value'] / fi_diffi_all['marginal_diffi_value'].sum())

    # Extract the first X elements from 'name_var' based on 'n_feats'
    fi_diffi_all['feat_selected'] = fi_diffi_all.apply(lambda row: list(fi_diffi_names['name_var'][:row['n_feats']]),
                                                       axis=1)

    fi_diffi_all = fi_diffi_all.drop(columns=['name_var', 'diffi_value', 'flag'], axis=1)

    return fi_diffi_all