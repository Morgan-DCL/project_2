import pandas as pd
import polars as pl
import numpy as np
import json
import os

# from numba import njit

from tools import (
    MyEncoder,
    logging,
    import_datasets,
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
    save: bool = False,
    data_type: list = "movie",
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

    first_df = import_datasets(
        sets["title_basics"],
        "polars",
        sep = "\t"
    )
    movies = first_df.filter(first_df["titleType"] == "movie")

    moviesO = movies.to_pandas()

    clean = DataCleaner()
    movies = clean.clean_porn(moviesO, columns_name="genres")
    logging.info(f"Cleaned : {len(moviesO) - len(movies)} rows")
    movies = pl.from_pandas(movies)

    dataframe = transform_raw_datas(
        'polars',
        "\t",
        sets["title_ratings"],
        sets["title_principals"],
        sets["name_basics"],
    )

    title_ratings = dataframe[0]
    title_principals = dataframe[1] # ici pour récuperer les nconst
    name_basics = dataframe[2] # ici pour avoir les acteurs

    logging.info(f"Joining first dataframes...")
    joined = movies.join(
        title_ratings,
        left_on = "tconst",
        right_on = "tconst"
    )
    if "movie" in data_type:
        logging.info(f"Renaming columns...")
        df = order_and_rename(
            joined,
            col_to_keep("movie"),
            col_renaming("movie")
        )
        df.write_csv(f"{folder_name}/movies.csv")

    if "actors" in data_type:
        logging.info(f"Joining second dataframes...")
        df_ = joined.join(
            title_principals,
            left_on = "tconst",
            right_on = "tconst"
        )

        actor_list = ["self", "actor", "actress"]
        condi = (df_["category"].is_in(actor_list))
        df_actors = df_.filter(condi)

        logging.info(f"Joining third dataframes...")
        df_actor1 = df_actors.join(
            name_basics,
            left_on = "nconst",
            right_on = "nconst"
        )

        logging.info(f"Renaming columns...")
        df_actor = order_and_rename(
            df_actor1,
            col_to_keep(""),
            col_renaming("")
        )
        df_actor.write_csv(f"{folder_name}/actors.csv")

    if "directors" in data_type:
        logging.info(f"Joining second dataframes...")
        df_ = joined.join(
            title_principals,
            left_on = "tconst",
            right_on = "tconst"
        )

        director_list = ["director"]
        condi = (df_["category"].is_in(director_list))
        df_directors = df_.filter(condi)

        logging.info(f"Joining third dataframes...")
        df_directors1 = df_directors.join(
            name_basics,
            left_on = "nconst",
            right_on = "nconst"
        )

        logging.info(f"Renaming columns...")
        df_director = order_and_rename(
            df_directors1,
            col_to_keep(""),
            col_renaming("")
        )
        df_director.write_csv(f"{folder_name}/directors.csv")

    # logging.info(f"Joining fourth dataframes...")
    # joined4 = joined3.join(
    #     imdb_title_akas,
    #     left_on = "tconst",
    #     right_on = "titleId"
    # )

    # logging.info(f"Renaming columns...")
    # df = order_and_rename(
    #     joined,
    #     col_to_keep(),
    #     col_renaming()
    # )

    # joined4 = joined4.to_pandas(strings_to_categorical=True)
    # print(joined4)
    # print(type(joined4))
    # joined5 = clean.fix_values(joined4, "fix_n")
    # df = clean.fix_values(joined5, "fix_encode")

    if save:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        logging.info("Saving dataframe...")
        df.write_csv(f"{folder_name}/{data_type}.csv")
        logging.info("Done!")


    return df


tvshows = ["tvShort", "tvSeries", "tvEpisode", "tvMiniSeries", "tvSpecial"]
# data__ = ["short", "tvMovie", "movie"]

df_to_create = ["movie", "actors", "directors"]

main_base_dataframe(save=False, data_type=df_to_create)



# def split_dataframe_category(
#         save: bool = False
# ):
#     """
#     Divise la dataframe créée par merging_dataframes en
#     4 parties différentes et sauvegarde les CSV.
#     """
#     logging.info("Loading the big boy df...")
#     if save:
#         big_data = main_base_dataframe(save=save)
#     else:
#         bigboy = "big_dataframe/big_dataframe.csv"
#         big_data = pl.read_csv(bigboy, ignore_errors=True)

#     tvshows = ["tvShort", "tvSeries", "tvEpisode", "tvMiniSeries", "tvSpecial"]

#     condi = (big_data["titre_type"].is_in(tvshows))

#     logging.info("Extracting TV Shows from DataFrame...")
#     tv_show = big_data.filter(condi)

#     logging.info("Extracting Shorts from DataFrame...")
#     short = big_data.filter(big_data["titre_type"] == "short")

#     logging.info("Extracting TV Movies from DataFrame...")
#     tv_movies = big_data.filter(big_data["titre_type"] == "tvMovie")

#     logging.info("Extracting Movies from DataFrame...")
#     movies = big_data.filter(big_data["titre_type"] == "movie")

#     logging.info("Sauvegarde des tableaux partitionnés...")
#     all_dfs = [
#         (tv_show, "tv_show.csv", "Writing tv_show..."),
#         (short, "short.csv", "Writing short..."),
#         (tv_movies, "tv_movies.csv", "Writing tv_movies..."),
#         (movies, "movies.csv", "Writing movies...")
#     ]

#     """
#     Création auto de plusieurs dataframe
#         - rating_movies
#         - rating_short
#         - rating_tv_show
#         - rating_tv_movies

#     Add colonne cuts à toutes les dataframes.
#         - avant 1980 par 20 ans
#         - > 1980 par 10 ans

#     Avant de sauvergarder, faire ne clean dans toutes les dataframes.
#     """

#     folder_name = f"clean_datasets"

#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)

#     for dfs in all_dfs:
#         logging.info(dfs[2])
#         dfs[0].write_csv(f"{folder_name}/{dfs[1]}")
#     logging.info("Done!")



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