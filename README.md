# Stability Index for Local Interpretability in Machine Learning

This project is focused on unsupervised models to identify anomalies in financial scenarios. Our approach not only enhances model performance but also ensures robust interpretability by integrating advanced machine learning techniques with SHAP (SHapley Additive exPlanations) values, fostering greater trust in the model's predictions.

## Key Features

* **Stability Index for Local Interpretability**: Introducing a new stability measure that quantifies the consistency of the model in ranking important features under data perturbations, crucial for building user trust.
* **Explainable AI (XAI) Models**: Enhancing transparency and understanding of model decisions through SHAP values.
* **Unsupervised Machine Learning**: Utilizing techniques like Isolation Forest for anomaly detection, crucial in identifying high-risk clients.
* **Robust Anomaly Detection**: Ensuring reliable and consistent predictions even under varying conditions.

## Project Structure

### Commit Strategy

To contribute, please follow the [contribution guide](https://github.com/edo1691/ad_shap_stability).

### Project Documentation

Comprehensive project documentation will be added soon. Stay tuned for detailed insights into our methodology and findings.

### Environment Setup

It's recommended to use a virtual environment for running this project to ensure consistency across different systems.

#### Data

For details on the dataset and variables used in this project, please refer to this [guide](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).

#### Virtual Environment Set-up

**Windows**
```bash
# Install environment
conda env create -p .\venv -f .\bin\local\environment.yml
# Activate the venv
conda activate \Users\eduardosepulveda\workspace_github_pers\ad_shap_stability\venv
```

**MacOS**
```bash
# Install environment
conda env create -p ./venv -f ./bin/local/environment.yml
# Activate the venv
conda activate /Users/allianz/workspace_github_pers/ad_shap_stability/venv

# Special libraries for mac M1
# conda install lightgbm
# pip install psycopg2
# pip install psycopg2-binary --force-reinstall --no-cache-dir
```

**Maintainer**
------------------

The developers responsible for the development and maintaining of this project.

* **Eduardo Sepúlveda** - *Author/Maintainer* - [eduardo.sepulveda@kuleuven.be](https://github.com/edo1691)
