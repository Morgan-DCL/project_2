import logging

import numpy as np
import pandas as pd
import polars as pl
from colored import attr, fg

logging.basicConfig(
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S')


def fix_N(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Effectue une opération de remplacement dans un DataFrame pandas.
    Remplace toutes les occurrences de '\\N' par 0.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame dans lequel effectuer le remplacement.

    Retourne
    -------
    pd.DataFrame
        Le DataFrame avec toutes les occurrences de '\\N' remplacées par 0.

    """
    return df.replace('\\N', 0)


def fix_encode_(
    column: str
) -> str:
    """

    Cette fonction prend une colonne sous forme de chaîne de caractères,
    l'encode en latin1, puis la décode en utf-8.
    Cela est souvent nécessaire lors de la manipulation
    de données qui ont été mal encodées.

    Paramètres
    ----------
    column : str
        La colonne à corriger. Doit être une chaîne de caractères.

    Retourne
    -------
    str
        La colonne avec l'encodage corrigé.

    """
    if isinstance(column, str):
        return column.encode('latin1').decode('utf-8')
    return column


def fix_encode_df(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Applique la correction d'encodage uniquement aux colonnes de
    type chaîne de caractères du DataFrame.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame dont les éléments doivent être corrigés.

    Retourne
    -------
    pd.DataFrame
        Le DataFrame avec l'encodage corrigé pour les colonnes
        de type chaîne de caractères.
    """
    for col in df.columns:
        df[col] = df[col].apply(fix_encode_)
    return df


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
            logging.info("Fixing N values...")
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
        logging.info(f"{fg('#ffa6c9')}{'🍆 ! Cleaning porn movies ! 🍆'}{attr(0)}")
        # logging.info("Cleaning porn movies...")
        datas = datas[datas[columns_name] != 0]
        msk = datas[columns_name].str.contains('Adult')
        return datas[~msk]

    def normalize_duration(self):
        pass

    def list_to_text(self):
        pass