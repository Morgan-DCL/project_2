import pandas as pd
import polars as pl
import numpy as np
import json
import os

# from numba import njit

from tools import (
    MyEncoder,
    logging,
    transform_raw_datas,
    order_and_rename,
    col_to_keep,
    col_renaming
)

from downloader import (
    downloader
)
from cleaner import (
    DataCleaner
)


def main_base_dataframe(
    download: bool = False,
    save: bool = False
) -> pl.DataFrame:

    """
    Création d'une grosse dataframe contenant exclusivement 3 dfs.
    -------------------------------------------------------------------------------

    name_basics :

        - nconst = ID unique de la personne
        - primaryName = Full name de la personne
        - birthYear = Année de Naissance
        - deathYear = Année de mort si existante, sinon \\N
        - primaryProfession = top 3 profession de la personne
        - knownForTitles = identifiant unique pour les films pour lesquels
            la personne est connue (on le retrouve dans les autres tables sauf
            titles.akas ou elle est = à titleId)
    -------------------------------------------------------------------------------

    title_basics :

        - tconst = identifiant unique pour chaque titre (titleId)
        - titleType = format du film (court-métrage, film, série, ...)
        - primaryTitle = titre utilisé par le directeur du film pour la promotion
        - originalTitle = titre original dans la VO
        - isAdult = 0 si le film est tout public, 1 si le film est un film pour adulte
        - startYear = année de sortie (ou date du premier épisode de la série)
        - endYear = date de la dernière saison de la série
        - runtimeMinutes = durée en minute
        - genres = catégorie (documentaire, animation, ...)
    -------------------------------------------------------------------------------

    title_principals:

        - tconst = identifiant unique pour le titre
        - ordering = idex pour les différentes personnes liées au titre
        - nconst = identifiant unique pour les personnes
        - category = catégorie du métier de la personne
        - job = intitulé du métier de la personne
        - characters = nom du personnage joué par la personne si acteur
    -------------------------------------------------------------------------------
    """
    if download:
        tsv_folder = "movies_datasets"
        if not os.path.exists(tsv_folder):
            os.makedirs(tsv_folder)
        downloader(tsv_folder)
    else:
        logging.info("Datasets ready to use!")

    with open("datasets_tsv.json", "r") as fp:
        sets = json.load(fp)

    dataframe = transform_raw_datas(
        'polars',
        sets["name_basics"],
        sets["title_basics"],
        sets["title_principals"]
    )

    imdb_name_basics = dataframe[0]
    imdb_titles_basics = dataframe[1]
    imdb_title_principals = dataframe[2]

    logging.info(f"Joining first dataframes...")
    joined = imdb_titles_basics.join(
        imdb_title_principals,
        left_on = "tconst",
        right_on = "tconst"
    )

    logging.info(f"Joining second dataframes...")
    joined2 = joined.join(
        imdb_name_basics,
        left_on = "nconst",
        right_on = "nconst"
    )

    logging.info(f"Renaming columns...")
    df = order_and_rename(
        joined2,
        col_to_keep(),
        col_renaming()
    )

    if save:
        folder_name = f"big_dataframe"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        logging.info("Saving dataframe...")
        df.write_csv(f"{folder_name}/big_dataframe.csv")
        logging.info("Done!")

    return df


def split_dataframe_category(
        save: bool = False
):
    """
    Divise la dataframe créée par merging_dataframes en
    4 parties différentes et sauvegarde les CSV.
    """
    logging.info("Loading the big boy df...")
    if save:
        big_data = main_base_dataframe(save=save)
    else:
        bigboy = "big_dataframe/big_dataframe.csv"
        big_data = pl.read_csv(bigboy, ignore_errors=True)  
 
    tvshows = ["tvShort", "tvSeries", "tvEpisode", "tvMiniSeries", "tvSpecial"]

    for tvshow in tvshows:
        condi = (big_data["titre_type"].is_in(tvshows))

    logging.info("Extracting TV Shows from DataFrame...")
    tv_show = big_data.filter(condi)

    logging.info("Extracting Shorts from DataFrame...")
    short = big_data.filter(big_data["titre_type"] == "short")

    logging.info("Extracting TV Movies from DataFrame...")
    tv_movies = big_data.filter(big_data["titre_type"] == "tvMovie")

    logging.info("Extracting Movies from DataFrame...")
    movies = big_data.filter(big_data["titre_type"] == "movie")

    logging.info("Sauvegarde des tableaux partitionnés...")
    all_dfs = [
        (tv_show, "tv_show.csv", "Writing tv_show..."),
        (short, "short.csv", "Writing short..."),
        (tv_movies, "tv_movies.csv", "Writing tv_movies..."),
        (movies, "movies.csv", "Writing movies...")
    ]

    folder_name = f"clean_datasets"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for dfs in all_dfs:
        logging.info(dfs[2])
        dfs[0].write_csv(f"{folder_name}/{dfs[1]}")
    logging.info("Done!")


split_dataframe_category()

# """
# First :
# Fixer et nettoyage des aberrations.
# Supprimer tous les films pornos !

# """
# cleaning = DataCleaner()
# test1 = cleaning.fix_values(df, "fix_encode")
# test = cleaning.fix_values(test1, "fix_n")
# clean_porn = cleaning.clean_porn(test)
# print(clean_porn.head())