import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.auxiliar.cleaner import Cleaner

# from libs.blob_connector import BlobConnector


class Sitb01:
    config = {
        "input_columns": [
            "NO_SIN",
            "C_CAU",
            "C_FAM_PROD",
            "D_SURV_SIN",
            "D_DCL",
            "IND_IRD",
            "NO_RSK",
            "C_NAT_SIN",
            "C_ET",
            "C_FORM",
            "D_DEB_EFF",
            "D_FIN_EFF",
            "M_TOT_SIN",
            "NB_GAR",
            "RESP_CIE",
            "IND_EXPE",
            "C_APPLN_MALUS",
            "NB_EXPE",
            "NB_HIST",
            "IND_GRAV",
            "IND_FORC_SIT",
            "C_SIT",
            "D_CL",
            "D_ROUVT",
            "NO_AVT",
            "NO_MAJ",
            "D_OUVT",
            "C_REVIS",
            "CNT_TY_GES",
            "M_RES_MORAT",
            "PROC_JUD",
            "IND_PMT_DIR",
            "C_ORGN_OUVT",
            "CPOST",
            "C_PERS_CLI",
            "IMM_SOC",
            "NO_CNT",
        ],
        "output_columns": [
            "NO_SIN",
            "C_CAU",
            "C_FAM_PROD",
            "D_SURV_SIN",
            "D_DCL",
            "NO_RSK",
            "C_NAT_SIN",
            "C_ET",
            "C_FORM",
            "D_DEB_EFF",
            "D_FIN_EFF",
            "M_TOT_SIN",
            "NB_GAR",
            "RESP_CIE",
            "IND_EXPE",
            "C_APPLN_MALUS",
            "NB_EXPE",
            "NB_HIST",
            "IND_FORC_SIT",
            "D_CL",
            "D_ROUVT",
            "NO_AVT",
            "NO_MAJ",
            "D_OUVT",
            "C_REVIS",
            "CNT_TY_GES",
            "M_RES_MORAT",
            "IND_PMT_DIR",
            "C_ORGN_OUVT",
            "CPOST",
            "weekday_surv_sin",
            "weekday_declaration",
            "d_surv_sin_is_weekend",
            "d_dcl_is_weekend",
            "reporting_delay_in_days",
            "contract_age_in_days",
            "is_closed",
            "is_serious_sinister",
            "has_judiciary_procedure",
            "IND_GRAV",
            "PROC_JUD",
            "C_PERS_CLI",
            "IMM_SOC",
            "NO_CNT",
        ],
        "column_types": {
            "int8_cols": [
                "PCT_REAS",
                "NB_DDR",
                "NB_EXPE",
                "NO_SIT_BREV_P",
                "NO_SIT_BREV_R",
                "NO_SIT_BREV_T",
                "NB_HIST",
                "NB_GAR",
                "NO_RSK",
            ],
            "int16_cols": ["C_DIAG", "C_GTA_CIE", "NO_M_DET"],
            "int32_cols": ["C_PERS_CLI"],
            "int64_cols": [],
            "float16_cols": [
                "NO_AVT",
                "NO_MAJ",
                "PCT_CED",
                "PCT_CIE",
                "NO_AVT_TERME",
                "NO_MAJ_TERME",
                "NO_AVT_TERME_",
                "NO_MAJ_TERME_",
            ],
            "float32_cols": ["M_TOT_SIN", "M_RES_MORAT"],
            "bool_cols": [
                "C_SOC",
                "IND_OPPO",
                "IND_EXPE",
                "IND_BLC",
                "C_REVIS",
                "IND_FORC_SIT",
                "FLG_MDF",
            ],
            "datetime_cols": ["TSTP"],
            "date_cols": [
                "D_SURV_SIN",
                "D_DCL",
                "D_OUVT",
                "D_CL",
                "D_ROUVT",
                "D_DEB_EFF",
                "D_FIN_EFF",
                "D_MAJ",
                "D_TRM_SIN",
                "D_MDP",
                "D_MDF_MALUS",
                "D_REVIS",
                "D_DEB_CHAN",
                "D_ENV_OUVT_AS",
                "D_ENV_CL_ASSU",
                "D_MDF_COAS",
                "D_DCL_EXT",
                "D_AFF_DCL",
                "D_TRM_SIN_SAU",
            ],
            "time_cols": ["H_SIN"],
            "str_cols": [
                "NO_SIN",
                "NO_CNT",
                "RUE",
                "VILLE",
                "CPOST",
                "NOM_PERS_LS",
                "IMM_SOC",
                "C_INTER_STATI",
                "REF_SIN_INTER",
                "NO_SIN_STD",
            ],
            "sparse_str_cols": ["PRN_PERS_LS", "DOS_COMMUN", "CIRC", "NO_DCL"],
            "category_cols": [
                "C_NAT_SIN",
                "C_SIT",
                "NO_SRSK",
                "C_PAYS",
                "C_DEV",
                "C_FAM_PROD",
                "C_FORM",
                "C_COAS",
                "NAT_COAS",
                "C_TIE_LS",
                "C_MTF_CL",
                "NO_M",
                "C_UTIL",
                "RESP_IDA",
                "CAS_IDA",
                "IND_FORC",
                "IND_IRD",
                "IND_IDA",
                "USER_BLC",
                "C_GESR_ORGN",
                "C_GESR_ACT",
                "C_CEN_GES",
                "NB_AA_ARCHIVE",
                "C_APPLN_MALUS",
                "C_CONFID",
                "BRM_RDR",
                "IND_RDR",
                "ACCUSE_RCP",
                "C_NAT_INTER",
                "C_NAT_STATI",
                "LIEU_ARCHIVE",
                "LIEU_GES",
                "CNT_TY_GES",
                "C_ARAG",
                "C_DEV_STK",
                "C_ENV_OUVT_AS",
                "C_ENV_CL_ASSU",
                "DOS_Y2000",
                "C_EVN_REAS",
                "C_MDF_COAS",
                "C_ORGN_DCL",
                "IND_GRAV",
                "PROC_JUD",
                "IND_PMT_DIR",
                "C_ORGN_OUVT",
                "C_APPLN_JOKER_1",
                "C_APPLN_JOKER_2",
                "C_SOC_ORGN",
                "C_ET",
                "RESP_CIE",
                "C_CEN_CMPTA",
                "IND_ACCEPT",
                "CYCLE_JOKER",
                "CYCLE_JOKER_S",
                "C_CAU",
            ],
        },
        "useless_columns": [
            "CPOST_ADV",
            "IMM",
            "NO_CNT_ADV",
            "NO_SIN_ADV",
            "C_PERS_EXPE",
            "NO_PERS_CDT",
            "TY_CNT",
            "TY_GES_AG",
            "C_PART",
            "D_CONSOL",
            "D_ENTER",
            "D_FIN_REV",
            "C_APPLN_JOKER",
            "C_APPLN_JOKER",
            "D_MDF",
            "C_SOC",
        ],
        # Why some columns are useless:
        # Except from C_SOC they are all containing only NaN values after cleaning
        # C_SOC is useless because it is a boolean that is only True (no False, no NaN)
        # 'date_format': '%Y%m%d',
        "encoding": "ISO-8859-1",
        "starting_date": datetime(2022, 1, 1),
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
                    prefix="CIRIARD/SITB01_columns"
                )
                data_file_path = self.blob_connector.load_latest_blob_locally(
                    run_date=self.run_date, prefix="CIRIARD/SITB01-"
                )
            else:
                columns_file_path = "/Users/allianz/workspace_github_pers/ad_shap_stability/test/data/inputs/SITB01_columns.csv"
                data_file_path = f"/Users/allianz/workspace_github_pers/ad_shap_stability/test/data/inputs/SITB01-{self.run_date}.csv"

            self.all_columns = pd.read_csv(columns_file_path)["Variable_name"].values
            self.column_names_cleaned = self.preprocess_column_names(
                columns=self.all_columns
            )
            self.filtered_column_types = self.filter_column_types(
                column_types=self.config["column_types"]
            )
            logging.info("Preprocessing Done")

            chunks = []

            for i, chunk in enumerate(
                pd.read_csv(
                    data_file_path,
                    sep="|",
                    names=self.column_names_cleaned,
                    usecols=self.config["input_columns"],
                    parse_dates=self.filtered_column_types["date_cols"],
                    # date_format=self.config['date_format'],
                    encoding=self.config["encoding"],
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

    def preprocess_column_names(self, columns):
        i = 1
        for index, column in enumerate(columns):
            if column == "C_APPLN_JOKER":
                columns[index] = f"{column}_{i}"
                i = i + 1
        return columns

    def filter_column_types(self, column_types):
        for col_type, cols in column_types.items():
            column_types[col_type] = [
                col for col in cols if col in self.config["input_columns"]
            ]
        return column_types

    def process_data(self, df):
        # df = self.flag_and_clean_wrong_formats(df)
        logging.info(f"Shape BEFORE cleaner {df.shape}")
        df = self.apply_cleaner(df)
        logging.info(f"Shape AFTER cleaner {df.shape}")
        df = self.apply_filters(df)
        df = self.create_features(df)
        df = self.fillna(df)
        return df

    def finalize_output(self):
        self.order_categories()
        self.filter_output()

    def flag_and_clean_wrong_formats(self, df):
        # Create new wrong_format_flag and wrong_format_detail columns
        df["wrong_format_flag"] = False
        df["wrong_format_detail"] = ""

        df = self.check_for_numeric_wrong_format(df)
        df = self.check_for_date_wrong_format(df)
        return df

    def check_for_numeric_wrong_format(self, df):
        for col in ["NO_SIN"]:
            is_numeric = df[col].str.isnumeric()
            df["wrong_format_flag"] = df["wrong_format_flag"] | ~is_numeric
            wrong_format_detail = np.where(
                is_numeric,
                "",
                f"Value of column {col}: " + df[col] + " | ",
            )
            df["wrong_format_detail"] = df["wrong_format_detail"] + wrong_format_detail
        return df

    def check_for_date_wrong_format(self, df):
        mandatory_date_columns = [
            "D_SURV_SIN",
            "D_DCL",
            "D_DEB_EFF",
            "D_FIN_EFF",
            "D_OUVT",
        ]
        columns_to_check = [
            col
            for col in mandatory_date_columns
            if col in self.filtered_column_types["date_cols"]
        ]
        for col in columns_to_check:
            is_not_a_date = df[col].isna()
            df["wrong_format_flag"] = df["wrong_format_flag"] | is_not_a_date
            wrong_format_detail = np.where(
                ~is_not_a_date,
                "",
                f"Value of column {col}: " + df[col].astype(str) + " | ",
            )
            df["wrong_format_detail"] = df["wrong_format_detail"] + wrong_format_detail
        return df

    def apply_cleaner(self, df):
        cleaner = Cleaner(**self.filtered_column_types)

        df = cleaner.clean_empty_strings(df)
        df = cleaner.clean_types(df)

        if "C_ET" in df.columns:
            df["C_ET"] = df["C_ET"].apply(lambda x: str(x) if pd.notnull(x) else x)
        return df

    def apply_filters(self, df):
        df = df[df["D_OUVT"] > self.config["starting_date"]].copy()
        logging.info(f"Shape after date filter {df.shape}")
        df = df[df["IND_IRD"] == "1"].copy()  # Motor claims only
        logging.info(f"Shape after IND_IRD filter {df.shape}")
        df = df[
            ~df["C_CAU"].isin(["74", "75", "76", "77", "78", "84", "RW", "ER", "SS"])
        ].copy()  # Remove causes business is not interested in
        logging.info(f"Shape after C_CAU filter {df.shape}")
        return df

    def create_features(self, df):
        df["weekday_surv_sin"] = df["D_SURV_SIN"].dt.weekday
        df["weekday_declaration"] = df["D_DCL"].dt.weekday
        df["d_surv_sin_is_weekend"] = df["weekday_surv_sin"].isin([5, 6])
        df["d_dcl_is_weekend"] = df["weekday_declaration"].isin([5, 6])
        df["reporting_delay_in_days"] = (df["D_DCL"] - df["D_SURV_SIN"]).dt.days
        df["contract_age_in_days"] = (df["D_SURV_SIN"] - df["D_DEB_EFF"]).dt.days
        # df['claim_age'] = (datetime.today() - df['D_OUVT']).dt.days
        # df['coverage_remaining_days'] = (df['D_FIN_EFF'] - df['D_SURV_SIN']).dt.days
        # df['reported_before_it_happened'] = df['reporting_delay_in_days'] < 0 # Not happening in our scope (Motor claims after 2010)
        df["is_closed"] = df["C_SIT"] == "C"
        df["has_been_reopened"] = df["D_ROUVT"].notna()
        df["is_serious_sinister"] = df["IND_GRAV"].notna()
        df["has_judiciary_procedure"] = df["PROC_JUD"].notna()
        return df

    def fillna(self, df):
        if "?" not in df["IND_PMT_DIR"].cat.categories:
            df["IND_PMT_DIR"] = df["IND_PMT_DIR"].cat.add_categories("?")
        df["IND_PMT_DIR"] = df["IND_PMT_DIR"].fillna("?")

        df["C_ORGN_OUVT"] = df["C_ORGN_OUVT"].cat.add_categories("?")
        df["C_ORGN_OUVT"] = df["C_ORGN_OUVT"].fillna("?")

        df["NO_RSK"] = df["NO_RSK"].fillna(0)

        df["IND_EXPE"] = df["IND_EXPE"].fillna(False)
        return df

    def order_categories(self):
        for col in self.filtered_column_types["category_cols"]:
            try:
                # Additional security in case the category type was lost by concatenation
                self.data[col] = self.data[col].astype("category")
                self.data[col] = self.data[col].cat.reorder_categories(
                    sorted(self.data[col].cat.categories)
                )
            except Exception as e:
                logging.info(f"Column {col} is of type: {self.data[col].dtype}")
                logging.info(f"Shape: {self.data[col].shape}")
                logging.info(f"Value counts: {self.data[col].value_counts().head(10)}")
                logging.info(f"Number of NaN: {self.data[col].isna().sum()}")
                raise e

    def filter_output(self):
        self.data = self.data[self.config["output_columns"]]
