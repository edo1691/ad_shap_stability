from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split
from src.stability.functions import stability_measure_model, stability_measure_shap


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
        X_train, X_test, y_train, y_test = train_test_split(data.drop(excluded, axis=1), data['y'], test_size=0.2,
                                                            random_state=42, stratify=data['y'])
    else:
        X_train, X_test = train_test_split(data.drop(excluded, axis=1), test_size=0.33, random_state=42)

    stab_model, _ = stability_measure_model(X_train, X_test, model,
                                            gamma=hyper['contamination'],
                                            unif=True,
                                            iterations=10)

    stab_shap, _ = stability_measure_shap(X_train, X_test, model,
                                          gamma=hyper['contamination'],
                                          unif=True,
                                          iterations=20)

    return iforest_report, conf_m, roc_auc, stab_model, stab_shap
