import ast
import json
import logging
import os
import re
from unicodedata import combining, normalize

import hjson
import numpy as np
import pandas as pd
import polars as pl
from colored import attr, fg
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super(MyEncoder, self).default(obj)


def import_config(config: str = "config.hjson", add: bool = False):
    conf_name = "../" + config if add else config
    with open(conf_name, "r") as fp:
        config = hjson.load(fp)
        if add:
            config["clean_df_path"] = "../" + config["clean_df_path"]
            config["download_path"] = "../" + config["download_path"]
            config["streamlit_path"] = "../" + config["streamlit_path"]
    return config


def make_filepath(filepath: str) -> str:
    """
    Crée un chemin de fichier si celui-ci n'existe pas déjà.

    Cette fonction vérifie si le chemin de fichier spécifié existe déjà.
    Si ce n'est pas le cas, elle crée le chemin de fichier.

    Paramètres
    ----------
    filepath : str
        Le chemin de fichier à créer.

    Retourne
    -------
    str
        Le chemin de fichier spécifié.

    Notes
    -----
    La fonction utilise la bibliothèque os pour interagir avec le système d'exploitation.
    """

    # dirpath = os.path.dirname(filepath) if filepath[-1] != "/" else filepath
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    return filepath


def hjson_dump(config: dict):
    with open("config/config.hjson", "w") as fp:
        hjson.dump(config, fp)