from src.stability.functions import stability_measure_model, local_stability_measure
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def metrics_iforest(df, model, stratify=True, random_state=42):
    excluded_cols = ['y', 'y_pred', 'prediction', 'y_scores', 'y_deci']
    cols_to_drop = [col for col in excluded_cols if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df['y']

    test_size = 0.4  # Define test size for splitting

    # Stratify split if required
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=stratify_param)

    if stratify:
        # Calculate metrics
        predictions = df.loc[X_test.index, 'prediction']  # Use prediction column for calculating metrics
        decision_scores = df.loc[X_test.index, 'y_deci']

        report = classification_report(y_test, predictions, target_names=['0', '1'], output_dict=True)
        roc_auc = roc_auc_score(y_test, decision_scores)

        # Stability Index
        stab_model, stab_model_list = stability_measure_model(X_train, X_test, model,
                                                              gamma=0.1,
                                                              unif=True,
                                                              iterations=5,
                                                              beta_flavor=2)
        # Local Interpretability Stability Index
        stab_shap, stab_shap_list = local_stability_measure(X_train, X_test, model,
                                                            gamma=0.5,
                                                            iterations=25,
                                                            beta_flavor=2)
    else:
        report = 0
        roc_auc = 0
        stab_model = 0
        stab_model_list = [0]
        stab_shap = 0
        stab_shap_list = [0]

    # Package all metrics into a dictionary
    metrics_dict = {
        'f1-score': report['1']['f1-score'],
        'recall': report['1']['recall'],
        'precision': report['1']['precision'],
        'roc_auc': roc_auc,
        'stab_model': stab_model,
        'stab_model_list': stab_model_list,
        'stab_shap': stab_shap,
        'stab_shap_list': stab_shap_list,
    }

    return metrics_dict
