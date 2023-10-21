import pandas as pd
import polars as pl
import numpy as np
import logging

from tools import (
    fix_N,
    fix_encode_df,
)

logging.basicConfig(
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S')


class DataCleaner():

    def __init__(
        self
    ) -> None:
        pass


    def fix_values(
        self,
        datas: pd.DataFrame,
        method: str = "fix_n",
    ):
        if method == "fix_n":
            logging.info("Fixing fucked up values...")
            return datas.apply(fix_N)
        elif method == "fix_encode":
            logging.info("Fixing encoding values...")
            return fix_encode_df(datas)
        else:
            raise ValueError(f"{method} not recognized!")

    def clean_porn(
        self,
        datas: pd.DataFrame,
        columns_name: str = "titre_genres"
    ):
        logging.info("Cleaning porn movies...")
        datas = datas[datas[columns_name] != 0]
        msk = datas[columns_name].str.contains('Adult')
        return datas[~msk]

    def normalize_duration(self):
        pass

    def list_to_text(self):
        pass