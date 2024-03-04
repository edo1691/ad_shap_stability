import numpy as np
import os
from scipy.io import loadmat
import pandas as pd
import scipy.io


def get_fs_dataset(dataset_id, path):
    if dataset_id == "arrhythmia" or dataset_id == "cardiotocography":
        data_root = os.path.join(path, "inputs", dataset_id + ".mat")
        mat = scipy.io.loadmat(data_root)
        mat = {k: v for k, v in mat.items() if k[0] != "_"}
        # Assuming df is your DataFrame
        # Create a sample DataFrame for illustration
        df = pd.DataFrame(mat["X"])

        # Rename columns using a loop
        for i in range(0, len(df.columns) + 1):
            old_col_name = i
            new_col_name = f"Col{i+1}"
            df.rename(columns={old_col_name: new_col_name}, inplace=True)

        y = pd.DataFrame(mat["y"])
        df["y"] = y

        y = np.array(df["y"])
    if dataset_id == "creditcard":
        data_root = os.path.join(path, "inputs", dataset_id + ".csv")
        df = pd.read_csv(data_root)
        df.rename(columns={'Class': 'y'}, inplace=True)
        df.drop(['Time', 'Amount'], axis=1, inplace=True)

    if dataset_id == "allianz":
        data_root = os.path.join(path, "inputs", dataset_id + ".parquet")
        df = pd.read_parquet(data_root)

    return df


def fs_datasets_hyperparams(dataset):
    data = {
        # arrhythmia
        ("arrhythmia"): {
            "contamination": 0.146,
            "max_samples": 256,
            "n_estimators": 100,
        },
        # Creditcard
        ("creditcard"): {
            "contamination": 0.05,
            "max_samples": 28,
            "n_estimators": 100,
        },
        # Creditcard
        ("allianz"): {
            "contamination": 0.1,
            "max_samples": 256,
            "n_estimators": 100,
        },
    }
    return data[dataset]