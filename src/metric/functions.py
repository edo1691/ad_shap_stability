from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split
from src.stability.functions import stability_measure_model, stability_measure_shap


def metrics_iforest(df, model, hyper, stratify=True, random_state=42):
    data = df.copy()
    excluded = ['y', 'y_pred', 'prediction', 'y_scores', 'y_deci']

    # Split the dataset into training and testing sets while stratifying based on the target variable
    if stratify:
        data_tr, data_te = train_test_split(data, test_size=0.4,
                                            random_state=random_state, stratify=data['y'])

        X_train = data_tr.drop(excluded, axis=1)
        X_test = data_te.drop(excluded, axis=1)
        y_train = data_tr[excluded]
        y_test = data_te[excluded]

        data_te_ad = data_te[data_te.y == 1]
        X_test_ad = data_te_ad.drop(excluded, axis=1)

        # Iforest report: precision, recall and f1_score
        report = classification_report(y_test['y'], y_test['prediction'], target_names=['0', '1'], output_dict=True)

        # Confusion matrix
        # conf_m = confusion_matrix(data['y'], data['prediction'])
        conf_m = 1

        fpr, tpr, thresholds = metrics.roc_curve(y_test['y'], y_test['y_deci'])
        roc_auc = metrics.auc(fpr, tpr)

        stab_shap_ad, _ = stability_measure_shap(X_train, X_test_ad, model,
                                                 gamma=0.1,
                                                 unif=True,
                                                 iterations=10,
                                                 beta_flavor=2)
    else:
        data_tr, data_te = train_test_split(data, test_size=0.4, random_state=random_state)

        X_train = data_tr.drop(excluded, axis=1)
        X_test = data_te.drop(excluded, axis=1)

        report = 0
        conf_m = 0
        roc_auc = 0

        stab_shap_ad = 0

    # Stability Index
    stab_model, stab_model_list = stability_measure_model(X_train, X_test, model,
                                            gamma=hyper['contamination'],
                                            unif=True,
                                            iterations=5,
                                            beta_flavor=2)

    stab_shap, stab_shap_list = stability_measure_shap(X_train, X_test, model,
                                          gamma=0.1,
                                          unif=True,
                                          iterations=5,
                                          beta_flavor=2)

    return report, conf_m, roc_auc, stab_model, stab_model_list, stab_shap_list, stab_shap, stab_shap_ad
