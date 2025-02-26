import numpy as np
import os
from scipy.io import loadmat
import pandas as pd
import scipy.io
from sklearn.preprocessing import OrdinalEncoder
from scipy.io import arff


def get_fs_dataset(dataset_id, path):

    if dataset_id == "example":
        n, ft_col = 100, 5
        df = np.random.randn(n, ft_col)  # Create the random data
        df = pd.DataFrame(df, columns=[f'feature_{i + 1}' for i in range(ft_col)])  # Convert to DataFrame
        # Add a new binary column 'y' with random 0 and 1 values
        df['y'] = np.random.randint(0, 2, size=n)

    if (dataset_id == "arrhythmia") | (dataset_id == "cardio") | (dataset_id == "musk") | (dataset_id == "mammography"):
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
        df = pd.read_csv(data_root, on_bad_lines='skip')
        df.rename(columns={'Class': 'y'}, inplace=True)
        df.drop(['Time', 'Amount'], axis=1, inplace=True)

    if dataset_id == "bank":
        data_root = os.path.join(path, "inputs", dataset_id + ".arff")
        data, meta = arff.loadarff(data_root)
        df = pd.DataFrame(data)
        df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        # Identify object columns
        object_columns = df.select_dtypes(include=['object']).columns
        numerical_cols = df.columns.difference(object_columns)
        # Fill NaN values with 0 for numerical columns
        df[object_columns] = df[object_columns].fillna('NaN_category')
        df[numerical_cols] = df[numerical_cols].fillna(0)
        # Initialize the ordinal encoder
        encoder = OrdinalEncoder()
        # Apply ordinal encoding to object columns
        df[object_columns] = encoder.fit_transform(df[object_columns])
        # df.rename(columns={'Class': 'y'}, inplace=True)
        # df.drop(['Time', 'Amount'], axis=1, inplace=True)

    if dataset_id == "german_data":
        data_root = os.path.join(path, "inputs", dataset_id + ".csv")
        df = pd.read_csv(data_root, on_bad_lines='skip')
        new_columns = {}
        for i, col in enumerate(df.columns):
            if i == len(df.columns) - 1:
                new_columns[col] = 'y'
            else:
                new_columns[col] = f'Col{i + 1}'

        df = df.rename(columns=new_columns)

    if dataset_id == "erp_fraud":
        data_root = os.path.join(path, "inputs", dataset_id + ".csv")
        df = pd.read_csv(data_root, on_bad_lines='skip')
        labels_mapping = {
            'NonFraud': 0,
            'Invoice_Kickback_II': 1,
            'Larceny_IV': 1,
            'Larceny_II': 1,
            'Larceny_III': 1,
            'Larceny_I': 1,
            'Invoice_Kickback_I': 1,
            'Corporate_Injury_I': 1
        }
        # Map the labels using the defined mapping
        df['Label'] = df['Label'].map(labels_mapping).fillna(0).astype(int)
        df.rename(columns={'Label': 'y'}, inplace=True)
        # Identify object columns
        object_columns = df.select_dtypes(include=['object']).columns
        numerical_cols = df.columns.difference(object_columns)
        # Fill NaN values with 0 for numerical columns
        df[object_columns] = df[object_columns].fillna('NaN_category')
        df[numerical_cols] = df[numerical_cols].fillna(0)
        # Initialize the ordinal encoder
        encoder = OrdinalEncoder()
        # Apply ordinal encoding to object columns
        df[object_columns] = encoder.fit_transform(df[object_columns])

    if dataset_id == "allianz":
        data_root = os.path.join(path, "inputs", dataset_id + ".parquet")
        df = pd.read_parquet(data_root)

    return df


def fs_datasets_hyperparams(dataset):
    data = {
        # example
        ("example"): {
            "contamination": 0.1,
            "max_samples": 'auto',
            "n_estimators": 100,
        },
        # arrhythmia
        ("arrhythmia"): {
            "contamination": 0.1,
            "max_samples": 'auto',
            "n_estimators": 100,
        },
        # mammography
        ("mammography"): {
            "contamination": 0.1,
            "max_samples": 256,
            "n_estimators": 100,
        },
        ("cardio"): {
            "contamination": 0.1,
            "max_samples": 256,
            "n_estimators": 100,
        },
        ("musk"): {
            "contamination": 0.1,
            "max_samples": 256,
            "n_estimators": 100,
        },
        ("bank"): {
            "contamination": 0.1,
            "max_samples": 256,
            "n_estimators": 100,
        },
        # Creditcard
        ("creditcard"): {
            "contamination": 0.1,
            "max_samples": 256,
            "n_estimators": 100,
        },
        # Creditcard
        ("allianz"): {
            "contamination": 0.1,
            "max_samples": 256,
            "n_estimators": 100,
        },
        # Creditcard
        ("erp_fraud"): {
            "contamination": 0.1,
            "max_samples": 256,
            "n_estimators": 100,
        },
    }
    return data[dataset]