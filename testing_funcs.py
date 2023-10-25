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
    create_main_movie_dataframe,
    single_base_transform,
    double_base_transform,
)

from downloader import (
    downloader
)

def main_base_dataframe(
    download: bool = False,
    data_type: list = ["movie"],
    folder_name: str = "big_dataframe"
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

    movies = create_main_movie_dataframe(sets)

    dataframe = transform_raw_datas(
        'polars',
        "\t",
        sets["title_ratings"],
        sets["title_principals"],
        sets["name_basics"],
    )

    title_ratings    = dataframe[0]
    title_principals = dataframe[1] # ici pour récuperer les nconst
    name_basics      = dataframe[2] # ici pour avoir les acteurs

    movies_ = single_base_transform(
        movies, title_ratings, "movies", folder_name, "tconst", "tconst"
    )

    if "actors" in data_type:
        name = "actors"
        actors = double_base_transform(
            movies_,
            title_principals,
            name_basics,
            name,
            ["self", "actor", "actress"],
            folder_name,
            "titre_id",
            "tconst",
            "nconst",
            "nconst",
        )

    if "directors" in data_type:
        name = "directors"
        directors = double_base_transform(
            movies_,
            title_principals,
            name_basics,
            name,
            ["director"],
            folder_name,
            "titre_id",
            "tconst",
            "nconst",
            "nconst",
        )

    # if save:
    #     if not os.path.exists(folder_name):
    #         os.makedirs(folder_name)

    #     logging.info("Saving dataframe...")
    #     df.write_csv(f"{folder_name}/{data_type}.csv")
    #     logging.info("Done!")


    # return df


df_to_create = ["movie", "actors", "directors"]
main_base_dataframe(data_type=df_to_create)

