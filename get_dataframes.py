import os
import pprint

import hjson
import numpy as np
import pandas as pd

from datetime import datetime

pd.set_option('display.float_format', lambda x: f'{x :.2f}')
import explo_data_analysis.eda_movies as eda
from cleaner import DataCleaner
from downloader import downloader
from tools import (
    create_main_movie_dataframe,
    decode_clean,
    decode_clean_actors,
    import_datasets,
    logging,
    single_base_transform,
    transform_raw_datas,
    make_filepath,
    order_and_rename_pandas,
    col_to_keep,
    col_renaming,
)


clean = DataCleaner()


class GetDataframes():
    def __init__(
        self,
        config: dict
    ):
        self.config = config
        self.default_path = make_filepath(config["clean_df_path"])
        self.tsv_data = config["data_sets_tsv"]

    def load_dataframe(
        self,
        path: str,
        funcs: callable
    ):
        name = path.split("/")[-1]
        if not os.path.exists(path):
            logging.info(f"File {name} not found. Creation...")
            return funcs()
        else:
            return import_datasets(path, "parquet")

    def get_movies_dataframe(self):
        name = "movies"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file):
            df = import_datasets(
                path_file,
                "parquet"
            )
        else:
            movies = create_main_movie_dataframe(
                self.config["data_sets_tsv"]
            )
            dataframe = transform_raw_datas(
                'polars',
                "\t",
                self.tsv_data["title_ratings"],
            )
            title_ratings = dataframe[0]

            df = single_base_transform(
                movies,
                title_ratings,
                "movies",
                self.default_path,
                "tconst",
                "tconst"
            )
            df = df.to_pandas()

            clean.fix_values(df, "fix_n")
            df['titre_date_sortie'].fillna(0, inplace=True)
            df['titre_date_sortie'] = df['titre_date_sortie'].astype("int64")
            df['titre_duree'] = df['titre_duree'].astype("int64")

            df_imdb = import_datasets(
                self.tsv_data["imdb_full"],
                "pandas"
            )
            merged = pd.merge(
                df,
                df_imdb,
                left_on="titre_id",
                right_on="imdb_id",
                how="left"
            )
            merged = merged.drop(eda.columns_to_drop_tmdb(), axis=1)
            max_ = merged.isna().sum()
            logging.info(f"Cleaned NaN Value : {max_.max()}")

            merged = merged.dropna()
            logging.info(f"Length dataframe merged with tmdb : {len(merged)}")

            col_list = ["spoken_languages", "production_countries"]
            merged = eda.clean_square_brackets(
                merged,
                col_list
            )
            merged = merged.dropna()
            logging.info(f"Length dataframe merged cleaned : {len(merged)}")

            merged = eda.apply_decode_and_split(
                merged,
                col_list,
                decode_clean
            )
            akas = import_datasets(
                self.tsv_data["title_akas"],
                types="pandas",
                sep="\t"
            )

            akas = akas[akas["region"] == 'FR']
            region_only = akas[["titleId", "region"]]

            logging.info("Merging tmdb and akas dataframes...")
            df = pd.merge(
                merged,
                region_only,
                left_on="titre_id",
                right_on="titleId"
            )

            logging.info("Drop all duplicated movies...")
            df.drop_duplicates(
                subset=["titre_id"], keep="first", inplace=True
            )
            condi = (
                df["status"] == "Released"
            )
            df = df[condi]
            df.drop(
                ["titleId"],
                inplace=True,
                axis=1
            )
            df = df.reset_index(drop="index")

            df = eda.split_columns(df, "titre_genres")
            df = eda.apply_decade_column(df)
            df = eda.drop_nan_values(df)

            df.to_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return df

    def get_persons_dataframes(self):
        name = "persons"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file):
            df = import_datasets(
                path_file,
                "parquet"
            )
        else:
            df = import_datasets(
                self.tsv_data["name_basics"],
                "pandas",
                sep="\t"
            )
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
            logging.info(f"Writing {name} dataframe...")
            df.to_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return df

    def get_characters_dataframe(self):
        name= "characters"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file):
            df = import_datasets(
                path_file,
                "parquet"
            )
        else:
            df = import_datasets(
                self.tsv_data["title_principals"],
                "pandas",
                sep="\t"
            )
            df.drop(["job"], inplace=True, axis=1)
            clean.fix_values(df, "fix_n")
            df["characters"] = np.where(
                df["characters"] == 0,
                "Unknown",
                df["characters"]
            )
            logging.info("Spliting...")
            df["characters"] = df["characters"].apply(
                decode_clean_actors
            ).str.split(",")
            df = df.reset_index(drop='index')
            logging.info(f"Writing {name} dataframe...")
            df.to_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return df

    def get_actors_dataframe(self):
        name = "actors"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file):
            df = import_datasets(
                path_file,
                "parquet"
            )
        else:
            df = self.load_dataframe(
                f"{self.default_path}/characters.parquet",
                self.get_characters_dataframe
            )
            logging.info(f"Get {name} only...")
            actors_list = ["self", "actor", "actress"]
            df = df[df['category'].isin(actors_list)]
            df = df.reset_index(drop='index')
            logging.info(f"Writing {name} dataframe...")
            df.to_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return df

    def get_directors_dataframe(self):
        name = "directors"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file):
            df = import_datasets(
                path_file,
                "parquet"
            )
        else:
            df = self.load_dataframe(
                f"{self.default_path}/characters.parquet",
                self.get_characters_dataframe
            )
            logging.info(f"Get {name} only...")
            df = df[df["category"].str.contains("director")]
            df = df.reset_index(drop='index')
            logging.info(f"Writing {name} dataframe...")
            df.to_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return df

    def get_actors_movies_dataframe(self):
        name = "actors_movies"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file):
            movies_actors = import_datasets(
                path_file,
                "parquet"
            )
        else:
            actors  = self.load_dataframe(
                f"{self.default_path}/actors.parquet",
                self.get_actors_dataframe
            )
            persons = self.load_dataframe(
                f"{self.default_path}/persons.parquet",
                self.get_persons_dataframes
            )
            movies  = self.load_dataframe(
                f"{self.default_path}/movies.parquet",
                self.get_movies_dataframe
            )

            actors_names = pd.merge(
                actors,
                persons,
                on = "nconst"
            )

            movies_actors = pd.merge(
                actors_names,
                movies,
                left_on = "tconst",
                right_on = "titre_id"
            )
            movies_actors.drop(
                ["tconst", "titleId"],
                inplace=True,
                axis=1
            )

            movies_actors = order_and_rename_pandas(
                movies_actors,
                col_to_keep(name),
                col_renaming(name)
            )

            movies_actors = movies_actors[col_renaming(name)]
            logging.info(f"Writing {name} dataframe...")
            movies_actors.to_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return movies_actors

    def get_directors_movies_dataframe(self):
        name = "directors_movies"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file):
            movies_directors = import_datasets(
                path_file,
                "parquet"
            )
        else:
            directors  = self.load_dataframe(
                f"{self.default_path}/directors.parquet",
                self.get_directors_dataframe
            )
            persons = self.load_dataframe(
                f"{self.default_path}/persons.parquet",
                self.get_persons_dataframes
            )
            movies  = self.load_dataframe(
                f"{self.default_path}/movies.parquet",
                self.get_movies_dataframe
            )

            directors_names = pd.merge(
                directors,
                persons,
                on = "nconst"
            )

            movies_directors = pd.merge(
                directors_names,
                movies,
                left_on = "tconst",
                right_on = "titre_id"
            )
            movies_directors.drop(
                ["tconst", "titleId"],
                inplace=True,
                axis=1
            )

            movies_directors = order_and_rename_pandas(
                movies_directors,
                col_to_keep(name),
                col_renaming(name)
            )

            movies_directors = movies_directors[col_renaming(name)]
            logging.info(f"Writing {name} dataframe...")
            movies_directors.to_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return movies_directors

    def get_dataframes(self, name: str):
        if name.lower() == "movies":
            return self.get_movies_dataframe()
        elif name.lower() == "persons":
            return self.get_persons_dataframes()
        elif name.lower() == "characters":
            return self.get_characters_dataframe()
        elif name.lower() == "actors":
            return self.get_actors_dataframe()
        elif name.lower() == "directors":
            return self.get_directors_dataframe()
        elif name.lower() == "actors_movies":
            return self.get_actors_movies_dataframe()
        elif name.lower() == "directors_movies":
            return self.get_directors_movies_dataframe()
        else:
            raise KeyError(f"{name.capitalize()} not know!")

def main(
    config: dict
):
    raise NotImplementedError
    data_sets_tsv = config.get("data_sets_tsv", {})
    for dataset_name, path in data_sets_tsv.items():
        name = path.split("/")[-1][:-4]
        if not os.path.exists(path) or config["download"]:
            logging.info(f"File {name} not found. Downloading...")
            # downloader(path)
        else:
            logging.info(f"File {name} already exist.")



# with open("config.hjson", "r") as fp:
#     config = hjson.load(fp)

# datas = GetDataframes(config)

# movies = datas.get_actors_movies_dataframe()
