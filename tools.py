import pandas as pd
import polars as pl
import numpy as np
# from numba import njit
import logging
import json

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
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


def transform_raw_datas(
    encryption: str = "polars",
    *datas: str
) -> list:
    """
    Transforme les données brutes en utilisant une méthode d'encryption spécifique.

    Paramètres
    ----------
    encryption : str, optional
        Le type d'encryption à utiliser pour transformer les données. Par défaut, "polars".
    *datas : str
        Les données brutes à transformer. Peut être plusieurs chaînes de caractères.

    Retourne
    -------
    list
        Une liste de données transformées.
        Chaque élément de la liste correspond à un ensemble de données transformé.

    """
    return (
        [
            import_datasets(data, types=encryption) for
            data in datas if data
        ]
    )



def import_datasets(
    datas: str,
    types: str,
    sep: str = "\t",
) -> pl.DataFrame:
    """
    Importe des ensembles de données à l'aide de pandas ou polars.

    Paramètres
    ----------
    datas : str
        Le chemin d'accès complet au fichier de données à importer.
    types : str
        Le type de bibliothèque à utiliser pour l'importation.
        Les options valides sont 'pandas' et 'polars'.
    sep : str, optionnel
        Le séparateur de colonnes à utiliser lors de l'importation du fichier.
        Par défaut, il s'agit d'une tabulation ('\t').

    Retourne
    -------
    DataFrame
        Un DataFrame contenant les données importées.

    Lève
    ----
    ValueError
        Si le type spécifié n'est ni 'pandas' ni 'polars'.

    """
    data_name = datas.split("/")[-1]
    if types == "pandas":
        logging.info(f"{types.capitalize()} loaded ! Importing {data_name}...")
        return pd.read_csv(datas, sep=sep, encoding="iso-8859-1")
    elif types == "polars":
        logging.info(f"{types.capitalize()} loaded ! Importing {data_name}...")
        return pl.read_csv(datas, separator=sep, ignore_errors=True)
    else:
        raise ValueError(
            f"{types} not recognize please use : [ pandas | polars ] ")


def order_and_rename(
    df: pl.DataFrame,
    og_col: list,
    new_col_name: list
):
    """
    Ordonne et renomme les colonnes dun DataFrame.

    Cette fonction prend un DataFrame,
    une liste de noms de colonnes originaux et une liste de nouveaux noms de colonnes.
    Elle renvoie un DataFrame avec les colonnes réorganisées et renommées.

    Paramètres
    ----------
    df : pl.DataFrame
        Le DataFrame dentrée sur lequel effectuer l'opération
        de réorganisation et de renommage.
    og_col : list
        Une liste de chaînes de caractères représentant
        les noms de colonnes originaux dans le DataFrame.
    new_col_name : list
        Une liste de chaînes de caractères représentant
        les nouveaux noms de colonnes pour le DataFrame.

    Retourne
    -------
    pl.DataFrame
        Un nouveau DataFrame avec les colonnes réorganisées et renommées.

    Remarques
    ---------
    Les listes og_col et new_col_name doivent avoir la même longueur.
    Chaque élément de og_col est associé à l'élément correspondant
    dans new_col_name pour le renommage.
    """
    return df.select(
        [
            pl.col(old).alias(new) for
            old, new in zip(og_col, new_col_name)
        ]
    )


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
    return column.encode('latin1').decode('utf-8')


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
        if df[col].dtype == 'O':
            df[col] = df[col].apply(fix_encode_)
    return df


def col_to_keep() -> list:
    """
    Renvoie une liste des noms de colonnes à conserver dans un dataframe.
    """
    return [
        "tconst",
        "primaryTitle",
        "titleType",
        "startYear",
        "endYear",
        "runtimeMinutes",
        "genres",
        "nconst",
        "primaryName",
        "birthYear",
        "category",
        "characters",
        "ordering"
    ]

def col_renaming() -> list:
    """
    Renvoie une liste des noms de colonnes à modifier dans un dataframe.
    """
    return [
        "titre_id",
        "titre_str",
        "titre_type",
        "titre_date_sortie",
        "titre_date_fin",
        "titre_duree",
        "titre_genres",
        "person_id",
        "person_name",
        "person_birthdate",
        "person_job",
        "person_role",
        "person_index"
    ]


def bins_generator(max_date_df: int) -> tuple:
    """
    Génère des intervalles de temps et leurs noms correspondants.

    Cette fonction crée des intervalles de temps à partir de l'année 1900
    jusqu'à une année maximale donnée.
    Les intervalles sont créés par tranches de 20 ans de 1900 à 1980,
    puis par tranches de 10 ans jusqu'à l'année maximale.
    Chaque intervalle est nommé par sa plage d'années correspondante.

    Paramètres
    ----------
    max_date_df : int
        L'année maximale à considérer pour la création des intervalles.

    Retourne
    -------
    tuple
        Un tuple contenant deux listes. La première liste contient
        les limites des intervalles de temps.
        La deuxième liste contient les noms correspondants de ces intervalles.

    """
    bins = [0, 1900]
    names = ["<1900"]

    for year in range(1900, 1980, 20):
        bins.append(year + 21)
        names.append(f"{year}-{year+20}")

    last_year = bins[-1]
    while last_year + 10 < int(max_date_df):
        bins.append(last_year + 10)
        names.append(f"{last_year-1}-{last_year+9}")
        last_year = bins[-1]

    bins.append(max_date_df)
    names.append(f">{last_year}")

    return bins, names
