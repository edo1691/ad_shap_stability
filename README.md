# Template to apply Machine Learning Models

Project to train predictive model to identify bad clients in credit risk.

The most important features are described bellow:

*  Template
*  Machine Learning
*  Statistical Learning
*  Classification Models
*  Regression Models

## Commit Strategy
------------------

If you want to contribute, follow this [guide](https://github.com/edo1691/Template_ML)

## Project Documentation
------------------------

TBD.

## Environment Settings
-----------------------

### Data

If you want to read the details of the dataset and variables, follow this [guide](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

### Virtual Environment Set-up

It is highly recommended setting up a virtual environment for the model pipeline developed in
Python language.

#### Windows
```bash

# Install environment
conda env create -p .\venv -f .\bin\local\environment.yml

# Activate the venv
conda activate \Users\eduardosepulveda\workspace_github_pers\Anomaly_detection\venv
```

#### MacOS
```bash

# Install environment
conda env create -p ./venv -f ./bin/local/environment.yml

# Activate the venv
conda activate /Users/allianz/workspace_github_pers/Anomaly_detection/venv

# Special libraries in mac M1

# conda install lightgbm
# pip install psycopg2
# pip install psycopg2-binary --force-reinstall --no-cache-dir
```

## Data Structure
-----------------
The directory structure is the following:

```markdown
    └── data
        ├── inputs 
        │   ├── data_db_201912.csv
        │   ├── data_db_202012.csv
        │   ├── ...
        │   ├── data_db_202012.pq
        │   └── data_db_202112.pq
        ├── intermediates
        │   ├── data_input_201912.pq
        │   ├── ...
        │   ├── data_input_202012.pq
        │   ├── ...
        │   └── data_input_pred_202201.pq
        ├── models
        │   ├── 20191231_{model_tgt}_{model_type}_select.txt
        │   ├── ...
        │   └── 20191231_{model_tgt}_{model_type}_model.pkl
        └── outputs
            ├── 202112_data_output_{YYYYMM_YYYYMM:train period}.pq
            ├── ...
            ├── 202112_data_output_{YYYYMM_YYYYMM:validatoin period}.pq
            ├── ...
            └── 202201_data_pred_output_{YYYYMM:prediction date}.pq  
```
Where /inputs is the general directory where all the raw files are stored, /models is where the backups
of the models, feature selection, etc... are stored.


## Maintainers
------------------

The developers responsible for the development and maintaining of this project.

* **Eduardo Sepúlveda** - *Author/Maintainer* - [eduardo.sepulveda.valdivia@gmail.com](https://github.com/edo1691)
