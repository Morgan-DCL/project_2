import os
import ast

import numpy as np
import pandas as pd

pd.set_option('display.float_format', lambda x: f'{x :.2f}')
import explo_data_analysis.eda_movies as eda
from cleaner import DataCleaner
from downloader import downloader
from tools import (
    col_renaming,
    col_to_keep,
    create_main_movie_dataframe,
    decode_clean,
    decode_clean_actors,
    import_datasets,
    logging,
    make_filepath,
    order_and_rename_pandas,
    single_base_transform,
    transform_raw_datas,
    get_tsv_files,
    replace_ids_with_titles,
    if_tt_remove,
)

clean = DataCleaner()


class GetDataframes():
    def __init__(
        self,
        config: dict
    ):
        self.config = config
        self.default_path = make_filepath(
            config["clean_df_path"])
        self.download_path = make_filepath(
            config["download_path"])
        self.tsv_file = get_tsv_files(
            self.download_path)

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

    def get_cleaned_movies(self, df: pd.DataFrame) -> pd.DataFrame:
        condi = (
            (df["titre_date_sortie"] >= self.config["movies_years"]) &
            (df["rating_avg"] >= self.config["movies_rating_avg"]) &
            ~(
                (df["titre_duree"] < self.config["movies_min_duration"]) |
                (df["titre_duree"] > self.config["movies_max_duration"])
            )
        )
        df = df[condi].reset_index(drop=True)
        return df

    def update_movies(self, path_file: str):
        movies_path = f"{self.default_path}/movies.parquet"
        df = import_datasets(movies_path, "parquet")
        df = self.get_cleaned_movies(df)
        df.to_parquet(path_file)
        return df


    def get_movies_dataframe(self, cleaned: bool = False):
        if not cleaned:
            name = "movies"
            path_file = f"{self.default_path}/{name}.parquet"

            if os.path.exists(path_file):
                df = import_datasets(
                    path_file,
                    "parquet"
                )
            else:
                movies = create_main_movie_dataframe(
                    self.tsv_file
                )
                dataframe = transform_raw_datas(
                    'polars',
                    "\t",
                    self.tsv_file["title_ratings"],
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
                    self.tsv_file["imdb_full"],
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
                logging.info(
                    f"Length dataframe merged with tmdb : {len(merged)}")

                col_list = ["spoken_languages", "production_countries"]
                merged = eda.clean_square_brackets(
                    merged,
                    col_list
                )
                merged = merged.dropna()
                logging.info(
                    f"Length dataframe merged cleaned : {len(merged)}")

                merged = eda.apply_decode_and_split(
                    merged,
                    col_list,
                    decode_clean
                )
                akas = import_datasets(
                    self.tsv_file["title_akas"],
                    types="pandas",
                    sep="\t"
                )

                akas = akas[akas["region"] == self.config["movies_region"]]
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
        else:
            name = "movies_cleaned"
            path_file = f"{self.default_path}/{name}.parquet"
            if os.path.exists(path_file):
                df = import_datasets(
                    path_file,
                    "parquet"
                )
                if self.check_if_moded(df):
                    logging.info("Values modified, creating new cleaned movies...")
                    df = self.update_movies(path_file)
                else:
                    logging.info("No need to update movies, all values are equals.")
            else:
                df = self.update_movies(path_file)
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
                self.tsv_file["name_basics"],
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
                self.tsv_file["title_principals"],
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

    def check_if_moded(self, df: pd.DataFrame) -> bool:
        check_year = df["titre_date_sortie"].min()
        check_min_duration = df["titre_duree"].min()
        check_max_duration = df["titre_duree"].max()
        check_rating = df["rating_avg"].min()
        return (
            check_year != self.config["movies_years"] or
            check_min_duration != self.config["movies_min_duration"] or
            check_max_duration != self.config["movies_max_duration"] or
            check_rating != self.config["movies_rating_avg"]
        )


    def get_actors_movies_dataframe(
        self, cleaned: bool = False, modify: bool = False
    ):
        name = "actors_movies"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file) and not modify:
            movies_actors = import_datasets(
                path_file,
                "parquet"
            )
            if self.check_if_moded(movies_actors):
                logging.info("Updating...")
                return self.get_actors_movies_dataframe(cleaned=cleaned, modify=True)
            else:
                logging.info(f"Dataframe {name} ready to use!")
                return movies_actors
        else:
            actors  = self.load_dataframe(
                f"{self.default_path}/actors.parquet",
                self.get_actors_dataframe
            )
            persons = self.load_dataframe(
                f"{self.default_path}/persons.parquet",
                self.get_persons_dataframes
            )
            if cleaned:
                movies  = self.load_dataframe(
                    f"{self.default_path}/movies_cleaned.parquet",
                    self.get_movies_dataframe(True)
                )
            else:
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
                ["tconst"],
                inplace=True,
                axis=1
            )

            movies_actors = order_and_rename_pandas(
                movies_actors,
                col_to_keep(name),
                col_renaming(name)
            )

            movies_actors = movies_actors[col_renaming(name)]
            logging.info("Replace tt by movies titles...")
            dict_titre = (
                movies_actors[['titre_id', 'titre_str']].drop_duplicates()
                .set_index('titre_id')
                .to_dict()['titre_str']
            )
            movies_actors['person_film'] = movies_actors['person_film'].apply(
                lambda x: if_tt_remove(replace_ids_with_titles(x, dict_titre))
            )

            logging.info(f"Writing {name} dataframe...")
            movies_actors.to_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return movies_actors

    def get_directors_movies_dataframe(
        self, modify: bool = False, cleaned: bool = False
    ):
        name = "directors_movies"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file) and not modify:
            movies_directors = import_datasets(
                path_file,
                "parquet"
            )
            if self.check_if_moded(movies_directors):
                logging.info("Updating...")
                return self.get_directors_movies_dataframe(cleaned=cleaned, modify=True)
            else:
                logging.info(f"Dataframe {name} ready to use!")
                return movies_directors
        else:
            directors  = self.load_dataframe(
                f"{self.default_path}/directors.parquet",
                self.get_directors_dataframe
            )
            persons = self.load_dataframe(
                f"{self.default_path}/persons.parquet",
                self.get_persons_dataframes
            )
            if cleaned:
                movies  = self.load_dataframe(
                    f"{self.default_path}/movies_cleaned.parquet",
                    self.get_movies_dataframe(True)
                )
            else:
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
                ["tconst"],
                inplace=True,
                axis=1
            )

            movies_directors = order_and_rename_pandas(
                movies_directors,
                col_to_keep(name),
                col_renaming(name)
            )

            movies_directors = movies_directors[col_renaming(name)]
            logging.info("Replace tt by movies titles...")
            dict_titre = (
                movies_directors[['titre_id', 'titre_str']].drop_duplicates()
                .set_index('titre_id')
                .to_dict()['titre_str']
            )
            movies_directors['person_film'] = movies_directors['person_film'].apply(
                lambda x: if_tt_remove(replace_ids_with_titles(x, dict_titre))
            )
            logging.info(f"Writing {name} dataframe...")
            movies_directors.to_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return movies_directors

    def get_dataframes(
        self, name: str, cleaned: bool = False
    ):
        downloader(self.config)
        if name.lower() == "movies":
            return self.get_movies_dataframe()
        elif name.lower() == "movies_cleaned":
            return self.get_movies_dataframe(cleaned)
        elif name.lower() == "persons":
            return self.get_persons_dataframes()
        elif name.lower() == "characters":
            return self.get_characters_dataframe()
        elif name.lower() == "actors":
            return self.get_actors_dataframe()
        elif name.lower() == "directors":
            return self.get_directors_dataframe()
        elif name.lower() == "actors_movies":
            return self.get_actors_movies_dataframe(cleaned=cleaned)
        elif name.lower() == "directors_movies":
            return self.get_directors_movies_dataframe(cleaned=cleaned)
        else:
            raise KeyError(f"{name.capitalize()} not know!")


# import hjson
# with open("config.hjson", "r") as fp:
#     config = hjson.load(fp)
# datas = GetDataframes(config)
# movies = datas.get_dataframes("movies")
# print(movies)

