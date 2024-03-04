import logging
import os
import sys

from src.load.functions import merge_data, load_source_data

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


class LoadProcessData:
    def __init__(self, conn_id: str = "", run_date: str = "", is_azure=True):
        self.data = None
        self.data_in = None
        self.data_out = None
        self.data01 = None
        self.data02 = None
        self.conn_id = conn_id
        self.run_date = run_date
        self.is_azure = is_azure
        self.df_encode = None
        self.df_raw = None
        self.process()

    def process(self):
        self.data01, self.data02 = load_source_data(
            conn_id=self.conn_id, run_date=self.run_date, is_azure=self.is_azure
        )
        self.data = merge_data(data01=self.data01, data02=self.data02)
