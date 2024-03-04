import logging
import os

import pandas as pd

from src.auxiliar.cleaner import Cleaner

# from libs.blob_connector import BlobConnector


class Sitb13:
    # useless columns are full of NaN values
    config = {
        "input_columns": ["NO-SIN", "C-NAT-INTVT"],
        "output_columns": ["NO-SIN", "witness_flag"],
        "column_types": {
            "int8_cols": [],
            "int16_cols": [],
            "int32_cols": [],
            "int64_cols": [],
            "float16_cols": [],
            "float32_cols": [],
            "bool_cols": [],
            "datetime_cols": [],
            "date_cols": [],
            "time_cols": [],
            "str_cols": [],
            "sparse_str_cols": [],
            "category_cols": ["C-SOC-ORGN", "C-NAT-INTVT"],
        },
        "useless_columns": [
            "NO-CMPT-SEQ",
            "QUAL-TIERS",
            "RGM-TVA",
            "C-BIC",
            "NO-TEL",
            "NO-PERMI",
            "TY-PERMI",
            "D-PERMI",
            "PAYS-PERMI",
            "CDT-HABIT",
            "TY-VEHIC",
            "ZONE-CHOC",
            "LIEU-INSP",
            "REG-GC",
            "C-NAT-GARG",
            "C-PERS-GARG",
            "C-IMMOB",
            "TY-VICT",
            "C-NAT-MED",
            "C-PERS-MED",
            "TY-BLES-1",
            "TY-BLES-2",
            "TY-BLES-3",
            "PCT-INC-P-1",
            "PCT-INC-P-2",
            "PCT-INC-P-3",
            "PCT-INC-P-4",
            "PCT-INC-P-5",
            "PCT-INC-P-6",
            "PCT-INC-P-7",
            "PCT-INC-P-8",
            "PCT-INC-P-9",
            "PCT-INC-P-10",
            "PCT-INV-P-1",
            "PCT-INV-P-2",
            "PCT-INV-P-3",
            "PCT-INV-P-4",
            "PCT-INV-P-5",
            "PCT-INV-P-6",
            "PCT-INV-P-7",
            "PCT-INV-P-8",
            "PCT-INV-P-9",
            "PCT-INV-P-10",
            "STAT-MED",
            "C-PART",
        ]
        # 'date_format': '%Y%m%d'
    }

    def __init__(self, conn_id: str = "", run_date: str = "", is_azure=True):
        self.conn_id = conn_id
        self.run_date = run_date
        self.is_azure = is_azure
        self.data = self.load_data()
        self.finalize_output()

    def load_data(self):
        columns_file_path = ""
        data_file_path = ""
        try:
            if self.is_azure:
                columns_file_path = self.blob_connector.load_latest_blob_locally(
                    prefix="CIRIARD/SITB13_columns"
                )
                data_file_path = self.blob_connector.load_latest_blob_locally(
                    run_date=self.run_date, prefix="CIRIARD/SITB13-"
                )
            else:
                columns_file_path = "/Users/allianz/workspace_github_pers/ad_shap_stability/test/data/inputs/SITB13_columns.csv"
                data_file_path = f"/Users/allianz/workspace_github_pers/ad_shap_stability/test/data/inputs/SITB13-{self.run_date}.csv"

                self.all_columns = pd.read_csv(columns_file_path)[
                    "Variable_name"
                ].values
                self.filtered_column_types = self.filter_column_types(
                    column_types=self.config["column_types"]
                )

            chunks = []

            for i, chunk in enumerate(
                pd.read_csv(
                    data_file_path,
                    sep="|",
                    names=self.all_columns,
                    usecols=self.config["input_columns"],
                    encoding="ISO-8859-1",
                    low_memory=False,
                    chunksize=50_000,
                )
            ):
                logging.info(f"Processing Chunk: {i + 1}")
                processed_chunk = self.process_data(chunk)
                logging.info(f"Processed Chunk Shape: {processed_chunk.shape}")
                if processed_chunk.shape[0] > 0:
                    chunks.append(processed_chunk)

            df = pd.concat(chunks)
            logging.info(f"Data Loaded in Pandas with Shape: {df.shape}")

        finally:
            # Delete the locally loaded source files
            if self.is_azure:
                for file_path in [columns_file_path, data_file_path]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logging.info(f"Successfully removed: {file_path}")
            else:
                pass

        return df

    def filter_column_types(self, column_types):
        for col_type, cols in column_types.items():
            column_types[col_type] = [
                col for col in cols if col in self.config["input_columns"]
            ]
        return column_types

    def process_data(self, df):
        df = self.apply_cleaner(df)
        df = self.apply_filters(df)
        df = self.create_features(df)
        return df

    def finalize_output(self):
        self.clean_output()
        self.filter_output()

    def apply_cleaner(self, df):
        cleaner = Cleaner(**self.filtered_column_types)

        df = cleaner.clean_empty_strings(df)
        df = cleaner.clean_types(df)
        return df

    def apply_filters(self, df):
        df = df[df["C-NAT-INTVT"] == "TE"]
        return df

    def create_features(self, df):
        df["witness_flag"] = df["C-NAT-INTVT"] == "TE"
        return df

    def clean_output(self):
        self.data = self.data.drop_duplicates()

    def filter_output(self):
        self.data = self.data[self.config["output_columns"]]
