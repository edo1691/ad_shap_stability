import gc

import numpy as np
import pandas as pd
from pandas import DataFrame


class Cleaner:
    def __init__(
        self,
        int8_cols: list,
        int16_cols: list,
        int32_cols: list,
        int64_cols: list,
        float16_cols: list,
        float32_cols: list,
        bool_cols: list,
        datetime_cols: list,
        date_cols: list,
        time_cols: list,
        str_cols: list,
        sparse_str_cols: list,
        category_cols: list,
    ):
        self.config = {
            "integers": {
                "function": self.cast_int,
                "args": [
                    {
                        "cols": int8_cols,
                        "without_na_type": "int8",
                        "with_na_type": "Int8",
                    },
                    {
                        "cols": int16_cols,
                        "without_na_type": "int16",
                        "with_na_type": "Int16",
                    },
                    {
                        "cols": int32_cols,
                        "without_na_type": "int32",
                        "with_na_type": "Int32",
                    },
                    {
                        "cols": int64_cols,
                        "without_na_type": "int64",
                        "with_na_type": "Int64",
                    },
                ],
            },
            "floats": {
                "function": self.simple_cast,
                "args": [
                    {"cols": float16_cols, "new_type": "float16"},
                    {"cols": float32_cols, "new_type": "float32"},
                ],
            },
            "boolean": {"function": self.cast_bool, "args": [{"cols": bool_cols}]},
            "category": {
                "function": self.cast_category,
                "args": [{"cols": category_cols}],
            },
            "sparse_str": {
                "function": self.simple_cast,
                "args": [{"cols": sparse_str_cols, "new_type": "Sparse[str]"}],
            },
            "datetime": {
                "function": self.cast_datetime,
                "args": [
                    {"cols": datetime_cols, "format_string": "%Y-%m-%d-%H.%M.%S.%f"}
                ],
            },
            "date": {
                "function": self.cast_date,
                "args": [{"cols": date_cols, "format_string": "%Y%m%d"}],
            },
            "time": {"function": self.cast_time, "args": [{"cols": time_cols}]},
        }

    def clean_empty_strings(self, df: DataFrame):
        for column in df.columns:
            if df[column].dtype == "object":
                df[column] = df[column].str.strip()
                gc.collect()
        # We have to choose between np.NaN and pd.NA
        df = df.replace("", np.NaN, regex=False)
        return df

    def clean_types(self, df: DataFrame):
        for col_type, col_config in self.config.items():
            for args_set in col_config["args"]:
                df = col_config["function"](df, **args_set)
        return df

    def cast_int(self, df, cols, without_na_type, with_na_type):
        for col in cols:
            new_type = without_na_type

            # Convert all values to float first to avoid TypeError
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Handle NaN values in source data with dedicated pandas IntegerArray type that casts np.NaN to pd.NA values
            if df[col].isna().sum() > 0:
                new_type = with_na_type

            print(f"Setting new type for column {col} as type {new_type}")
            df[col] = df[col].astype(new_type)
        return df

    def cast_category(self, df, cols):
        # df[cols] = df[cols].astype('str').astype('category')
        df[cols] = df[cols].astype("category")
        for col in cols:
            df[col].cat.categories = df[col].cat.categories.astype("str")
        return df

    def simple_cast(self, df, cols, new_type):
        df[cols] = df[cols].astype(new_type)
        return df

    def cast_bool(self, df, cols):
        bool_mapper = {0: False, 1: True, "0": False, "1": True, "N": False, "Y": True}

        # Use 'boolean' type: the pandas dedicated BooleanArray data type to keep np.NaN as pd.NA.
        # Use 'bool' dtype to remove NaN (NaN will be set as True by default)
        for col in cols:
            df[col] = df[col].map(bool_mapper).astype("boolean")
        return df

    def cast_datetime(self, df, cols, format_string):
        for col in cols:
            df[col] = pd.to_datetime(df[col], format=format_string, errors="raise")
        return df

    def cast_date(self, df, cols, format_string):
        for col in cols:
            df[col] = pd.to_datetime(df[col], format=format_string, errors="coerce")
        return df

    def cast_time(self, df, cols):
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int16")
            df[col] = df[col].where(cond=(df[col] <= 2400), other=0)
            # df[f'{col}_enc_cos'] = df[col].apply(encode_hours_cos).astype('float16')
            # df[f'{col}_enc_sin'] = df[col].apply(encode_hours_sin).astype('float16')
        return df
