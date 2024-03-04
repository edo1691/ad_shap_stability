import logging
import os
import sys
from pandas import DataFrame

from src.auxiliar.sitb01 import Sitb01
from src.auxiliar.sitb13 import Sitb13

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def load_source_data(conn_id, run_date, is_azure):
    logging.info("LOADING STIB01")
    data01 = Sitb01(conn_id=conn_id, run_date=run_date, is_azure=is_azure)
    logging.info("LOADING STIB13")
    data02 = Sitb13(conn_id=conn_id, run_date=run_date, is_azure=is_azure)
    return data01, data02


def merge_data(data01: DataFrame, data02: DataFrame):
    data = data01.data.merge(
        data02.data, how="left", left_on="NO_SIN", right_on="NO-SIN"
    )
    data["witness_flag"].fillna(False, inplace=True)
    data.drop("NO-SIN", inplace=True, axis=1)
    data.set_index(["NO_SIN"], drop=True, inplace=True)
    # We transform float16 to float32 because of parquet format
    var = ["NO_AVT", "NO_MAJ"]
    data[var] = data[var].astype("float32", errors="ignore")
    return data
