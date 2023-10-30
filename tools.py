import json
# from numba import njit
import logging
import os
import ast

import numpy as np
import pandas as pd
import polars as pl

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


def make_filepath(filepath: str) -> str:
    # dirpath = os.path.dirname(filepath) if filepath[-1] != "/" else filepath
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    return filepath


def get_download_link():
    return {
        "name_basics" :     "https://datasets.imdbws.com/name.basics.tsv.gz",
        "title_akas" :      "https://datasets.imdbws.com/title.akas.tsv.gz",
        "title_basics" :    "https://datasets.imdbws.com/title.basics.tsv.gz",
        "title_episode" :   "https://datasets.imdbws.com/title.episode.tsv.gz",
        "title_principals": "https://datasets.imdbws.com/title.principals.tsv.gz",
        "title_ratings" :   "https://datasets.imdbws.com/title.ratings.tsv.gz"
    }


def get_tsv_files(folder_name: str) -> dict:
    return {
        "name_basics" :      f"{folder_name}/name_basics.tsv",
        "title_ratings" :    f"{folder_name}/title_ratings.tsv",
        "title_akas" :       f"{folder_name}/title_akas.tsv",
        "title_basics" :     f"{folder_name}/title_basics.tsv",
        "title_episode" :    f"{folder_name}/title_episode.tsv",
        "title_principals" : f"{folder_name}/title_principals.tsv",
        "imdb_full" :        f"{folder_name}/tmdb_full.csv",
    }

def replace_ids_with_titles(id_list: str, dict_titre: dict) -> list:
    if isinstance(id_list, str):
        id_list = ast.literal_eval(id_list)
    return [dict_titre.get(titre_id, titre_id) for titre_id in id_list]

def if_tt_remove(id_list):
    return [t for t in id_list if not t.startswith("tt")]

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

def order_and_rename_pandas(
    df: pd.DataFrame,
    og_col: list,
    new_col_name: list
) -> pd.DataFrame:
    """
    Ordonne et renomme les colonnes d'un DataFrame.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entrée.
    datasets : str
        Type de données pour choisir les colonnes à conserver.
    new_col_name : list
        Liste des nouveaux noms de colonnes.

    Retourne
    -------
    pd.DataFrame
        DataFrame avec les colonnes réorganisées et renommées.

    Remarques
    ---------
    Les listes col_to_keep(datasets) et new_col_name doivent avoir la même longueur.
    """
    rename_dict = {old: new for old, new in zip(og_col, new_col_name)}
    df.rename(columns=rename_dict, inplace=True)
    return df


def col_to_keep(
    datasets: str
) -> list:
    """
    Renvoie une liste des noms de colonnes à conserver dans un dataframe.
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
    Renvoie une liste des noms de colonnes à modifier dans un dataframe.
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

    Parameters
    ----------
    sets : dict
        Un dictionnaire contenant les ensembles de données à importer. La clé doit être "title_basics".

    Returns
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
    Paramètres
    ----------
    data1 : pl.DataFrame
        Premier dataframe à fusionner.
    data2 : pl.DataFrame
        Deuxième dataframe à fusionner.
    left : str, optionnel
        Nom de la colonne sur laquelle effectuer la fusion dans le premier dataframe.
        Par défaut, c'est 'tconst'.
    right : str, optionnel
        Nom de la colonne sur laquelle effectuer la fusion dans le deuxième dataframe.
        Par défaut, c'est 'tconst'.

    Retourne
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

    Paramètres
    ----------
    data : pl.DataFrame
        Le DataFrame à filtrer.
    filter_list : list
        La liste des valeurs à utiliser pour le filtrage.
    column_to_filter : str, optionnel
        Le nom de la colonne à filtrer. Par défaut, il s'agit de "category".

    Retourne
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


def create_persons_dataframe(
    link: str
) -> pd.DataFrame:
    """
    Crée un DataFrame pandas à partir d'un lien donné, nettoie et modifie les types de données.

    Cette fonction importe un ensemble de données à partir d'un lien, supprime certaines colonnes,
    corrige les valeurs, modifie les types de données et enregistre le DataFrame résultant en format parquet.

    Parameters
    ----------
    link : str
        Le lien vers l'ensemble de données à importer.

    Returns
    -------
    pd.DataFrame
        Le DataFrame nettoyé et modifié.

    Notes
    -----
    Les colonnes 'deathYear' et 'primaryProfession' sont supprimées du DataFrame.
    Les valeurs de la colonne 'knownForTitles' sont divisées en plusieurs lignes.
    Le type de données de la colonne 'birthYear' est modifié en 'int64'.
    Le DataFrame est enregistré en format parquet sous le nom 'clean_datasets/person.parquet'.
    """

    df = import_datasets(link, "pandas", sep="\t")
    df.drop(["deathYear", "primaryProfession"], axis=1, inplace=True)
    clean.fix_values(df, "fix_n")
    logging.info("Spliting and modifing dtypes...")
    df["knownForTitles"] = np.where(
        df["knownForTitles"] == 0,
        "Unknown",
        df["knownForTitles"]
    )
    df["knownForTitles"] = df["knownForTitles"].str.split(",")
    df["birthYear"] = df["birthYear"].astype("int64")
    df = df.reset_index(drop='index')
    logging.info("Writing persons dataframe...")
    df.to_parquet("clean_datasets/personfffffff.parquet")


def create_specific_dataframe(
    link: str,
    job: str,
    create: bool,
):
    """
    Créer et enregistrer une dataframe specifique:
        - actors
        - directors
        - writers
        - compositors

    Parameters
    ----------
    link : str
        lien du fichier a récuperer
    actors_df : bool, optional
        _description_, by default True
    directors_df : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    df = import_datasets(link, "pandas", sep="\t")
    df.drop(["job"], inplace=True, axis=1)
    clean.fix_values(df, "fix_n")
    df["characters"] = np.where(
        df["characters"] == 0,
        "Unknown",
        df["characters"]
    )
    df["characters"] = df["characters"].apply(decode_clean_actors).str.split(",")

    if job == "actors":
        actors_list = ["self", "actor", "actress"]
        actors = df[df['category'].isin(actors_list)]
        actors = actors.reset_index(drop='index')
        logging.info("Writing actors dataframe...")
        actors.to_parquet("clean_datasets/actors.parquet")

    if job == "directors":
        directors = df[df["category"].str.contains("director")]
        directors = directors.reset_index(drop='index')
        logging.info("Writing directors dataframe...")
        directors.to_parquet("clean_datasets/directors.parquet")

    if job == "writers":
        raise NotImplementedError

    if job == "compositors":
        raise NotImplementedError

def create_actors_and_directors_dataframe(
    link: str,
    prepare_d: bool = True,
    actors_df: bool = True,
    directors_df: bool = True,
):
    """
    Crée des dataframes pour les acteurs et les réalisateurs à partir d'un lien donné.

    Paramètres
    ----------
    link : str
        Le lien vers le jeu de données à importer.
    actors_df : bool, optionnel
        Si True (par défaut), un dataframe pour les acteurs est créé.
    directors_df : bool, optionnel
        Si True (par défaut), un dataframe pour les réalisateurs est créé.

    Notes
    -----
    Cette fonction importe un jeu de données, supprime la colonne 'job', nettoie les valeurs,
    décode et nettoie les noms des acteurs, puis crée des dataframes pour les acteurs et les réalisateurs
    si demandé. Les dataframes sont ensuite enregistrés en format parquet.

    La colonne 'characters' est remplacée par 'Unknown' si elle contient la valeur 0.
    Les acteurs sont identifiés par les catégories 'self', 'actor' et 'actress'.
    Les réalisateurs sont identifiés par la présence du mot 'director' dans la catégorie.
    """

    df = import_datasets(link, "pandas", sep="\t")
    df.drop(["job"], inplace=True, axis=1)
    clean.fix_values(df, "fix_n")
    df["characters"] = np.where(
        df["characters"] == 0,
        "Unknown",
        df["characters"]
    )
    df["characters"] = df["characters"].apply(decode_clean_actors).str.split(",")
    return df
    if actors_df:
        actors_list = ["self", "actor", "actress"]
        actors = df[df['category'].isin(actors_list)]
        actors = actors.reset_index(drop='index')
        logging.info("Writing actors dataframe...")
        actors.to_parquet("clean_datasets/actors.parquet")
    if directors_df:
        directors = df[df["category"].str.contains("director")]
        directors = directors.reset_index(drop='index')
        logging.info("Writing directors dataframe...")
        directors.to_parquet("clean_datasets/directors.parquet")



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
    df : pl.DataFrame
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
    return (
        serie.replace("[", "")
            .replace("]", "")
            .replace('"', "")
            .replace("'", "")
        )