from src.stability.functions import local_stability_measure

import logging
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_model_experiment(fi_shap_all, df, hyper, gamma=0.146, iterations=10,
                         n_estimators_list=None, seed=42, dataset_id=None):
    """
    Optimized version of the run_model_experiment function to improve performance.
    """
    if n_estimators_list is None:
        n_estimators_list = [1, 5, 25]

    metrics_list = []

    # Pre-split the target variable
    y = df['y']

    # Pre-split the dataset into training and testing sets (used consistently)
    xtr_all, xte_all, ytr_all, yte_all = train_test_split(df.drop(columns=['y']), y, test_size=0.1, random_state=seed)

    # Iterate over each set of selected features
    for idx, row in fi_shap_all.iterrows():
        selected_features = row['feat_selected']
        n_features = len(selected_features)
        n_features_cum_shap_percentage = row['cum_value_percentage']

        # Log the start of processing for the current number of features
        logging.info(f'Starting experiment with {n_features} features.')

        # Use only selected features
        xtr = xtr_all[selected_features]
        xte = xte_all[selected_features]

        # Loop over each n_estimators value
        for n_estimators in n_estimators_list:
            # Log the start of processing for the current number of estimators
            logging.info(f'Starting model training with {n_features} features and {n_estimators} estimators.')

            # Update the hyperparameter dictionary with the current n_estimators
            hyper['n_estimators'] = n_estimators

            # Initialize and train the IsolationForest model with the current n_estimators
            model = IsolationForest(**hyper, random_state=seed)
            model.fit(xtr)

            # Prediction and scoring
            y_pred = model.predict(xtr)
            y_scores = -model.score_samples(xtr)
            y_decision = -model.decision_function(xtr)

            # Vectorized creation of predictions DataFrame
            y_pred = (y_pred == -1).astype(int)
            df_predictions = pd.DataFrame({
                'y_pred': y_pred,
                'y_scores': y_scores,
                'y_decision': y_decision,
                'y_real': ytr_all[xtr.index]
            })

            roc_auc = roc_auc_score(df_predictions['y_real'], df_predictions['y_decision'])

            report = classification_report(df_predictions['y_real'], df_predictions['y_pred'],
                                           target_names=['Normal', 'Anomaly'], output_dict=True)

            # Call the stability measure function
            local_scores, local_scores_list, ranking = local_stability_measure(
                xtr, xte, model, gamma, iterations=iterations,
                psi=0.8, beta_flavor=2, subset_low=0.25, subset_high=0.75, rank_method=True
            )

            # Collect metrics for this iteration
            metrics_list.append({
                'dataset_id': dataset_id,
                'n_feat': n_features,
                'n_features_cum_shap_percentage': n_features_cum_shap_percentage,
                'n_estimators': n_estimators,
                'f1-score': report['Anomaly']['f1-score'],
                'recall': report['Anomaly']['recall'],
                'precision': report['Anomaly']['precision'],
                'roc_auc': roc_auc,
                'smli': local_scores.mean(),
                'smli_all': local_scores
            })

    # Convert the metrics list to a DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    return metrics_df
