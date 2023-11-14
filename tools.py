import ast
import json
# from numba import njit
import logging
import os
import re

import hjson
import numpy as np
import pandas as pd
import polars as pl
from colored import attr, fg
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from cleaner import DataCleaner

clean = DataCleaner()

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


def import_config(config: str ="../config/config.hjson"):
    with open(config, "r") as fp:
        return hjson.load(fp)


def make_filepath(filepath: str) -> str:
    """
    Crée un chemin de fichier si celui-ci n'existe pas déjà.

    Cette fonction vérifie si le chemin de fichier spécifié existe déjà. Si ce n'est pas le cas, elle crée le chemin de fichier.

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
    with open("config.hjson", "w") as fp:
        hjson.dump(config, fp)


def get_download_link() -> dict:
    """
    Renvoie un dictionnaire contenant les noms des bases de données IMDB en
    tant que clés et leurs liens de téléchargement respectifs en tant que valeurs.

    Returns
    -------
    dict
        Un dictionnaire où les clés sont les noms des bases de
        données IMDB
        'name_basics',
        'title_akas',
        'title_basics',
        'title_episode',
        'title_principals',
        'title_ratings'
        et les valeurs sont les liens de téléchargement correspondants.
    """

    return {
        "name_basics" :     "https://datasets.imdbws.com/name.basics.tsv.gz",
        "title_akas" :      "https://datasets.imdbws.com/title.akas.tsv.gz",
        "title_basics" :    "https://datasets.imdbws.com/title.basics.tsv.gz",
        "title_episode" :   "https://datasets.imdbws.com/title.episode.tsv.gz",
        "title_principals": "https://datasets.imdbws.com/title.principals.tsv.gz",
        "title_ratings" :   "https://datasets.imdbws.com/title.ratings.tsv.gz"
    }


def get_tsv_files(folder_name: str) -> dict:
    """
    Obtient les chemins des fichiers TSV dans un dossier spécifique.

    Paramètres
    ----------
    folder_name : str
        Le nom du dossier contenant les fichiers TSV.

    Retourne
    -------
    dict
        Un dictionnaire contenant les noms des fichiers TSV
        comme clés et leurs chemins respectifs comme valeurs.

    """
    return {
        "name_basics" : f"{folder_name}/name_basics.tsv",
        "title_ratings" : f"{folder_name}/title_ratings.tsv",
        "title_akas" : f"{folder_name}/title_akas.tsv",
        "title_basics" : f"{folder_name}/title_basics.tsv",
        "title_episode" : f"{folder_name}/title_episode.tsv",
        "title_principals" : f"{folder_name}/title_principals.tsv",
        "imdb_full" : f"clean_datasets/tmdb_updated.parquet",
    }

def replace_ids_with_titles(
    id_list: str,
    dict_titre: dict
) -> list:
    """
    Remplace les identifiants dans une liste par leurs titres correspondants à partir d'un dictionnaire.

    Paramètres
    ----------
    id_list : str
        Une chaîne de caractères représentant une liste d'identifiants.
        Les identifiants doivent être séparés par des virgules et la liste doit être entourée de crochets.

    dict_titre : dict
        Un dictionnaire où les clés sont les identifiants et les valeurs sont les titres correspondants.

    Retourne
    -------
    list
        Une liste où chaque identifiant de la liste d'entrée a été remplacé par
        son titre correspondant dans le dictionnaire.
        Si un identifiant ne se trouve pas dans le dictionnaire,
        il est laissé tel quel dans la liste de sortie.
    """
    if isinstance(id_list, str):
        id_list = ast.literal_eval(id_list)
    return [
        dict_titre.get(titre_id, titre_id)
        for titre_id in id_list
    ]

def if_tt_remove(id_list: list) -> list:
    """
    Effectue une opération de filtrage sur une liste d'identifiants,
    en supprimant ceux qui commencent par "tt".

    Paramètres
    ----------
    id_list : list
        Une liste de chaînes de caractères représentant les identifiants à filtrer.

    Retourne
    -------
    list
        Une nouvelle liste contenant uniquement les identifiants qui ne commencent pas par "tt".

    """

    return [
        t for t in id_list
        if not t.startswith("tt")
    ]

def transform_raw_datas(
    encryption: str = "polars",
    sep: str = ",",
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
            import_datasets(data, types=encryption, sep=sep) for
            data in datas if data
        ]
    )


def import_datasets(
    datas: str,
    types: str,
    sep: str = ",",
) -> pd.DataFrame:
    """
    Importe des ensembles de données à l'aide de pandas ou polars.

    Parameters
    ----------
    datas : str
        Le chemin d'accès complet au fichier de données à importer.
    types : str
        Le type de bibliothèque à utiliser pour l'importation.
        Les options valides sont 'pandas', 'parquet' et 'polars'.
    sep : str, optional
        Le séparateur de colonnes à utiliser lors de l'importation du fichier.
        Par défaut, il s'agit d'une virgule (',').

    Returns
    -------
    pl.DataFrame
        Un DataFrame contenant les données importées.

    Raises
    ------
    ValueError
        Si le type spécifié n'est ni 'pandas', ni 'parquet', ni 'polars'.
    """
    data_name = datas.split("/")[-1]
    if types == "pandas":
        # logging.info(f"{fg('#ffa6c9')}{'🍆 ! Cleaning porn movies ! 🍆'}{attr(0)}")
        logging.info(f"{types.capitalize()} loaded ! Importing {data_name[:-4]}...")
        return pd.read_csv(datas, sep=sep, low_memory=False) #, encoding="iso-8859-1"
    if types == "parquet":
        # logging.info(f"{fg('#ffa6c9')}{'🍆 ! Cleaning porn movies ! 🍆'}{attr(0)}")
        logging.info(f"{types.capitalize()} loaded ! Importing {data_name[:-8]}...")
        return pd.read_parquet(datas)
    elif types == "polars":
        logging.info(f"{types.capitalize()} loaded ! Importing {data_name[:-4]}...")
        return pl.read_csv(datas, separator=sep, ignore_errors=True)
    else:
        raise ValueError(
            f"{types} not recognize please use : [ pandas | polars ] ")


def order_and_rename(
    df: pl.DataFrame,
    og_col: list,
    new_col_name: list
) -> pd.DataFrame:
    """
    Ordonne et renomme les colonnes d'un DataFrame.

    Cette fonction prend un DataFrame, une liste de noms de colonnes originaux et une liste de nouveaux noms de colonnes.
    Elle renvoie un DataFrame avec les colonnes réorganisées et renommées.

    Parameters
    ----------
    df : pl.DataFrame
        Le DataFrame d'entrée sur lequel effectuer l'opération de réorganisation et de renommage.
    og_col : list
        Une liste de chaînes de caractères représentant les noms de colonnes originaux dans le DataFrame.
    new_col_name : list
        Une liste de chaînes de caractères représentant les nouveaux noms de colonnes pour le DataFrame.

    Returns
    -------
    pl.DataFrame
        Un nouveau DataFrame avec les colonnes réorganisées et renommées.

    Notes
    -----
    Les listes og_col et new_col_name doivent avoir la même longueur. Chaque élément de og_col est associé à l'élément correspondant
    dans new_col_name pour le renommage.
    """
    return df.select(
        [
            pl.col(old).alias(new) for
            old, new in zip(og_col, new_col_name)
        ]
    )

def order_and_rename_pandas(
    df: pd.DataFrame,
    og_col: list,
    new_col_name: list
) -> pd.DataFrame:
    """
    Ordonne et renomme les colonnes d'un DataFrame pandas.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entrée sur lequel effectuer les opérations.
    og_col : list
        Liste des noms originaux des colonnes à renommer.
    new_col_name : list
        Liste des nouveaux noms de colonnes.

    Retourne
    -------
    pd.DataFrame
        DataFrame avec les colonnes réorganisées et renommées.

    Notes
    -----
    Les listes og_col et new_col_name doivent avoir la même longueur.
    """
    rename_dict = {old: new for old, new in zip(og_col, new_col_name)}
    df.rename(columns=rename_dict, inplace=True)
    return df


def col_to_keep(
    datasets: str
) -> list:
    """
    Renvoie une liste des noms de colonnes à conserver dans un dataframe en fonction du type de données.

    Parameters
    ----------
    datasets : str
        Le type de données pour lequel obtenir les noms de colonnes.
        Les valeurs valides sont "movies", "actors",
        "directors", "actors_movies" et "directors_movies".

    Returns
    -------
    list
        Une liste des noms de colonnes à conserver.

    Raises
    ------
    KeyError
        Si le type de données n'est pas valide.
    """
    if datasets == "movies":
        return [
            "tconst",
            "primaryTitle",
            "startYear",
            "runtimeMinutes",
            "genres",
            "averageRating",
            "numVotes",
        ]
    if datasets in ["actors", "directors"]:
        return [
            # "titre_id",
            # "titre_str",
            # "titre_date_sortie",
            # "titre_duree",
            # "titre_genres",
            # "rating_avg",
            # "rating_votes",
            "nconst", # name_basics             "person_id",
            "primaryName", # name_basics        "person_name",
            "birthYear", # name_basics          "person_birthdate",
            "category", # name_basics           "person_job",
            # "characters", # name_basicsa        "person_role",
            # "ordering", # name_basics           "person_index",
            "knownForTitles", # name_basics     "person_film",
            "ordering", # name_basics     "person_film",
        ]
    if datasets in ["actors_movies", "directors_movies"]:
        return [
            "titre_id",
            "titre_str",
            "titre_date_sortie",
            "titre_duree",
            "titre_genres",
            "rating_avg",
            "rating_votes",
            "original_language",
            "original_title",
            "popularity",
            "production_countries",
            "revenue",
            "spoken_languages",
            "status",
            "region",
            "cuts",
            "nconst", # name_basics             "person_id",
            "primaryName", # name_basics        "person_name",
            "birthYear", # name_basics          "person_birthdate",
            "category", # name_basics           "person_job",
            "characters", # name_basicsa        "person_role",
            "knownForTitles", # name_basics     "person_film",
            "ordering", # name_basics           "person_film",
        ]
    else:
        raise KeyError(f"{datasets} n'est pas valide!")


def col_renaming(
    datasets: str
) -> list:
    """
    Fonction pour renvoyer une liste de noms de colonnes à modifier dans un dataframe.

    Paramètres
    ----------
    datasets : str
        Le nom du dataset pour lequel la liste des noms de colonnes est requise.

    Retourne
    -------
    list
        Une liste de noms de colonnes à modifier.
        Si le dataset est "movies", la liste contient les noms de colonnes
        spécifiques à ce dataset.
        Si le dataset est "actors_movies" ou "directors_movies", la liste contient les noms de
        colonnes spécifiques à ces datasets. Si le dataset n'est pas reconnu, une KeyError est levée.

    Lève
    -----
    KeyError
        Si le nom du dataset n'est pas reconnu.
    """
    if datasets == "movies":
        return [
            "titre_id",
            "titre_str",
            "titre_date_sortie",
            "titre_duree",
            "titre_genres",
            "rating_avg",
            "rating_votes",
        ]
    if datasets in ["actors_movies", "directors_movies"]:
        return [
            "titre_id",
            "titre_str",
            "titre_date_sortie",
            "titre_duree",
            "titre_genres",
            "rating_avg",
            "rating_votes",
            "original_language",
            "original_title",
            "popularity",
            "production_countries",
            "revenue",
            "spoken_languages",
            "status",
            "region",
            "cuts",
            "person_id",
            "person_name",
            "person_birthdate",
            "person_job",
            "person_role",
            "person_film",
            "person_index",
        ]
    else:
        raise KeyError(f"{datasets} n'est pas valide!")


def bins_generator(max_date_df: int) -> tuple:
    """
    Génère des intervalles de temps et leurs noms correspondants.

    Paramètres
    ----------
    max_date_df : int
        L'année maximale à considérer pour la génération des intervalles.

    Retourne
    -------
    tuple
        Un tuple contenant deux listes. La première liste contient les limites des intervalles de temps.
        La deuxième liste contient les noms correspondants à ces intervalles.

    """
    bins = [0, 1900]
    names = ["<1900"]

    for year in range(1900, max_date_df, 10):
        bins.append(year + 9)
        names.append(f"{year}-{year+9}")

    if max_date_df >= bins[-2]:
        names[-1] = f">{names[-1][:4]}"

    return bins, names


def create_main_movie_dataframe(
    sets: dict,
) -> pl.DataFrame:
    """
    Crée un DataFrame principal pour les films à partir d'un ensemble de données spécifié.

    Cette fonction importe d'abord les ensembles de données, filtre les films, nettoie les films pornographiques,
    puis convertit le DataFrame en Polars pour fusionner.

    Paramètres
    ----------
    sets : dict
        Un dictionnaire contenant les ensembles de données à importer. La clé doit être "title_basics".

    Renvoie
    -------
    pl.DataFrame
        Un DataFrame Polars contenant les informations des films nettoyés.

    """
    first_df = import_datasets(
        sets["title_basics"],
        "polars",
        sep = "\t"
    )
    movies = first_df.filter(
        first_df["titleType"] == "movie"
    )
    # Convert into Pandas df to clean porn movies
    moviesO = movies.to_pandas()

    # Clean porn movies
    movies = clean.clean_porn(moviesO, columns_name="genres")
    logging.info(f"Cleaned : {len(moviesO) - len(movies)} rows")

    # Convert back into Polars to proceed merging
    movies = pl.from_pandas(movies)
    return movies


def join_dataframes(
    data1: pl.DataFrame,
    data2: pl.DataFrame,
    left: str = 'tconst',
    right: str = 'tconst',
) -> pl.DataFrame:
    """
    Fusionne deux dataframes sur la base des colonnes spécifiées.

    Parameters
    ----------
    data1 : pl.DataFrame
        Le premier dataframe à fusionner.
    data2 : pl.DataFrame
        Le deuxième dataframe à fusionner.
    left : str, optional
        Le nom de la colonne sur laquelle effectuer la fusion dans le premier dataframe.
        Par défaut, c'est 'tconst'.
    right : str, optional
        Le nom de la colonne sur laquelle effectuer la fusion dans le deuxième dataframe.
        Par défaut, c'est 'tconst'.

    Returns
    -------
    pl.DataFrame
        Un nouveau dataframe qui est le résultat de la fusion des deux dataframes d'entrée.
    """
    return data1.join(data2, left_on=left, right_on=right)


def filter_before_join(
    data: pl.DataFrame,
    filter_list: list,
    column_to_filter: str = "category"
) -> pl.DataFrame:
    """
    Filtre les données d'un DataFrame en fonction d'une liste de filtres et d'une colonne spécifique.

    Parameters
    ----------
    data : pl.DataFrame
        Le DataFrame à filtrer.
    filter_list : list
        La liste des valeurs à utiliser pour le filtrage.
    column_to_filter : str, optional
        Le nom de la colonne à filtrer. Par défaut, il s'agit de "category".

    Returns
    -------
    pl.DataFrame
        Le DataFrame filtré.
    """
    condi = (data[column_to_filter].is_in(filter_list))
    return data.filter(condi)


def single_base_transform(
    datas1: pl.DataFrame,
    datas2: pl.DataFrame,
    name: str = "movies",
    folder_name: str = "big_dataframe",
    left_on: str = "tconst",
    right_on: str = "tconst",
) -> pl.DataFrame:
    """
    Effectue une transformation de base unique sur deux DataFrames pandas, les joint,
    les renomme et les enregistre en CSV.

    Paramètres
    ----------
    datas1 : pandas.DataFrame
        Premier DataFrame à transformer.
    datas2 : pandas.DataFrame
        Deuxième DataFrame à transformer.
    name : str, optionnel
        Nom de la transformation, par défaut "movies".
    folder_name : str, optionnel
        Nom du dossier où le fichier CSV sera enregistré, par défaut "big_dataframe".
    left_on : str, optionnel
        Nom de la colonne sur laquelle effectuer la jointure à gauche, par défaut "tconst".
    right_on : str, optionnel
        Nom de la colonne sur laquelle effectuer la jointure à droite, par défaut "tconst".

    Retourne
    -------
    pandas.DataFrame
        DataFrame transformé, joint, renommé et enregistré en CSV.

    """
    logging.info(f"Joining {name} dataframes...")
    df_ = join_dataframes(
        datas1,
        datas2,
        left_on,
        right_on
    )
    logging.info(f"Renaming {name} columns...")
    df = order_and_rename(
        df_,
        col_to_keep(name),
        col_renaming(name)
    )
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)

    # df.write_parquet(f"{folder_name}/{name}.parquet")
    return df


def double_base_transform(
    datas1: pl.DataFrame,
    datas2: pl.DataFrame,
    datas3: pl.DataFrame,
    name: str = "actors",
    filter_list: list = [],
    folder_name: str = "big_dataframe",
    left1: str = "tconst",
    right1: str = "tconst",
    left2: str = "nconst",
    right2: str = "nconst",
) -> pl.DataFrame:
    """
    Effectue une double transformation de base sur les données fournies.

    Paramètres
    ----------
    datas1 : pl.DataFrame
        Premier jeu de données à transformer.
    datas2 : pl.DataFrame
        Deuxième jeu de données à transformer.
    datas3 : pl.DataFrame
        Troisième jeu de données à transformer.
    name : str, optionnel
        Nom associé aux données, par défaut "actors".
    filter_list : list, optionnel
        Liste des filtres à appliquer avant la jointure, par défaut [].
    folder_name : str, optionnel
        Nom du dossier où les données transformées seront stockées, par défaut "big_dataframe".
    left1 : str, optionnel
        Nom de la colonne à utiliser comme clé gauche pour la première jointure, par défaut "tconst".
    right1 : str, optionnel
        Nom de la colonne à utiliser comme clé droite pour la première jointure, par défaut "tconst".
    left2 : str, optionnel
        Nom de la colonne à utiliser comme clé gauche pour la deuxième jointure, par défaut "nconst".
    right2 : str, optionnel
        Nom de la colonne à utiliser comme clé droite pour la deuxième jointure, par défaut "nconst".

    Retourne
    -------
    pl.DataFrame
        DataFrame résultant de la double transformation de base.
    """
    logging.info(f"Joining {name} dataframes...")
    df__ = join_dataframes(
        datas1,
        datas2,
        left1,
        right1
    )

    df_ = filter_before_join(
        df__, filter_list, "category"
    )

    df = single_base_transform(
        df_,
        datas3,
        name,
        folder_name,
        left2,
        right2
    )
    return df



def decode_clean(
    serie: pd.Series
) -> str:
    """
    Décode et nettoie une série pandas.

    Cette fonction prend une série pandas en entrée et retourne une chaîne de caractères
    où certains caractères spécifiques sont supprimés. Les caractères supprimés sont :
    "[", "]", "'", " ", et '"'.

    Parameters
    ----------
    serie : pd.Series
        La série pandas à décoder et nettoyer.

    Returns
    -------
    str
        La série pandas décodée et nettoyée, sous forme de chaîne de caractères.
    """
    return (
        serie.replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace(" ", "")
            .replace('"', "")
        )


def decode_clean_actors(
    serie: pd.Series
) -> str:
    """
    Décode et nettoie une série d'acteurs.

    Cette fonction prend une série pandas en entrée, supprime tous les caractères non désirés tels que les crochets, les guillemets doubles et simples, et renvoie la série nettoyée sous forme de chaîne de caractères.

    Paramètres
    ----------
    serie : pd.Series
        La série pandas contenant les noms d'acteurs à nettoyer.

    Retourne
    -------
    str
        La série nettoyée sous forme de chaîne de caractères.

    """
    return (
        serie.replace("[", "")
            .replace("]", "")
            .replace('"', "")
            .replace("'", "")
        )

def clean_overview(
    text: str
) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def full_lower(
    text: str
):
    return text.replace(" ", "")

def color(
    text: str,
    color: str = None
) -> str:
    if color and color.startswith("#"):
        return f"{fg(color)}{text}{attr(0)}"
    elif color == 'red':
        return f"{fg(1)}{text}{attr(0)}"
    elif color == 'green':
        return f"{fg(2)}{text}{attr(0)}"
    elif color == 'yellow':
        return f"{fg(3)}{text}{attr(0)}"
    elif color == 'blue':
        return f"{fg(4)}{text}{attr(0)}"
    else:
        return text

def check_titre(
    df: pd.DataFrame, string: str, max: int = 1
) -> pd.DataFrame:
    """
    Sélectionne et renvoie les lignes d'un DataFrame correspondant à un critère.

    Cette fonction filtre les lignes du DataFrame basé sur la présence d'une
    chaîne de caractères spécifique dans les colonnes 'titre_id' ou 'titre_clean'.
    Si la chaîne commence par "tt", la recherche s'effectue dans 'titre_id'.
    Sinon, elle se fait dans 'titre_clean'. Seules les 'max' premières lignes
    correspondantes sont retournées.

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame dans lequel effectuer la recherche.
    string : str
        La chaîne de caractères à rechercher.
    max : int, optional
        Le nombre maximal de lignes à retourner (par défaut 1).

    Returns
    -------
    pd.DataFrame
        Un DataFrame contenant les lignes correspondant au critère de recherche,
        limité au nombre spécifié par 'max'.
    """
    string = string.lower()
    if string.startswith("tt"):
        return df[df["titre_id"].str.contains(string)][:max]
    else:
        string = string.replace(" ", "").replace("-", "").replace("'", "").replace(":", "").lower()
        return df[df["titre_clean"].str.contains(string)][:max]