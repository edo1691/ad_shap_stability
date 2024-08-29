import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, classification_report
from src.stability.functions import local_stability_measure


def run_model_experiment(fi_shap_all, df, hyper, gamma=0.146, iterations=10,
                         n_estimators_list=[1, 5, 25], seed=42, dataset_id=None):
    """
    Runs an experiment using the IsolationForest model on different subsets of features and
    n_estimators values.

    Parameters:
        fi_shap_all (DataFrame): DataFrame containing the selected features for each iteration.
        df (DataFrame): The main dataset containing features and the target column 'y'.
        hyper (dict): Dictionary containing hyperparameters for the IsolationForest model.
        gamma (float): Gamma value for the local stability measure function.
        iterations (int): Number of iterations for the local stability measure function.
        n_estimators_list (list): List of n_estimators values to try.
        seed (int): Random seed for reproducibility.
        local_stability_measure (function): A function to calculate local stability measure.
        dataset_id (str or int): Identifier for the dataset being used.

    Returns:
        DataFrame: DataFrame containing metrics for each combination of selected features and n_estimators.
    """

    metrics_list = []

    # Loop over each set of selected features
    for idx, row in fi_shap_all.iterrows():
        selected_features = row['feat_selected']
        n_features = len(selected_features)
        n_features_cum_shap_percentage = row['cum_value_percentage']

        # Split the DataFrame into features (X) and target (y) using the selected features
        X = df[selected_features]  # Use only selected features
        y = df['y']  # Target (the 'y' column)

        # Loop over each n_estimators value
        for n_estimators in n_estimators_list:
            # Update the hyperparameter dictionary with the current n_estimators
            hyper['n_estimators'] = n_estimators

            # Split into training and testing sets
            xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=seed)

            # Initialize and train the IsolationForest model with the current n_estimators
            model = IsolationForest(**hyper, random_state=seed)
            model.fit(xtr)

            # Prediction
            y_pred = model.predict(xtr)
            y_scores = -model.score_samples(xtr)
            y_decision = -model.decision_function(xtr)

            # Create a new DataFrame with the same index as xtr
            df_predictions = pd.DataFrame({
                'y_pred': y_pred,
                'y_scores': y_scores,
                'y_decision': y_decision,
                'y_real': ytr
            }, index=xtr.index)

            df_predictions['y_pred'] = df_predictions['y_pred'].apply(lambda x: 1 if x == -1 else 0)

            roc_auc = roc_auc_score(df_predictions.y_real, df_predictions.y_decision)

            report = classification_report(df_predictions.y_real, df_predictions.y_pred,
                                           target_names=['Normal', 'Anomaly'], output_dict=True)

            # Call your stability measure function
            local_scores, local_scores_list, ranking = local_stability_measure(
                xtr,
                xte,
                model,
                gamma,
                iterations=iterations,
                psi=0.8,
                beta_flavor=2,  # pick from: 1, 2
                subset_low=0.25,
                subset_high=0.75,
                rank_method=True
            )

            # Collect metrics for this iteration
            metrics_dict = {
                'dataset_id': dataset_id,
                'n_feat': n_features,  # Number of features used
                'n_features_cum_shap_percentage': n_features_cum_shap_percentage,
                'n_estimators': n_estimators,  # Number of estimators used in this iteration
                'f1-score': report['Anomaly']['f1-score'],
                'recall': report['Anomaly']['recall'],
                'precision': report['Anomaly']['precision'],
                'roc_auc': roc_auc,
                'smli': local_scores.mean(),
                'smli_all': local_scores
            }

            metrics_list.append(metrics_dict)

    # Convert the metrics list to a DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    return metrics_df