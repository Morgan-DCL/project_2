import os
import ast

import numpy as np
import pandas as pd
import pprint

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
    clean_overview,
    full_lower,
    hjson_dump,
    color
)

clean = DataCleaner()


class GetDataframes():
    """
    Classe pour gérer et manipuler les DataFrames.

    Attributs
    ----------
    config : dict
        Dictionnaire de configuration.
    default_path : str
        Chemin par défaut pour stocker les fichiers.
    download_path : str
        Chemin pour télécharger les fichiers.
    tsv_file : str
        Fichier TSV à traiter.

    Méthodes
    -------
    load_dataframe(path: str, funcs: callable) -> pd.DataFrame:
        Charge un DataFrame à partir d'un fichier ou le crée si le fichier n'existe pas.

    get_cleaned_movies(df: pd.DataFrame) -> pd.DataFrame:
        Nettoie le DataFrame des films en fonction des critères de configuration.

    update_movies(path_file: str) -> pd.DataFrame:
        Met à jour le DataFrame des films.

    get_movies_dataframe(cleaned: bool = False) -> pd.DataFrame:
        Récupère le DataFrame des films, nettoyé ou non.

    get_persons_dataframes() -> pd.DataFrame:
        Récupère le DataFrame des personnes.

    get_characters_dataframe() -> pd.DataFrame:
        Récupère le DataFrame des personnages.

    get_actors_dataframe() -> pd.DataFrame:
        Récupère le DataFrame des acteurs.

    get_directors_dataframe() -> pd.DataFrame:
        Récupère le DataFrame des réalisateurs.

    check_if_moded(df: pd.DataFrame) -> bool:
        Vérifie si le DataFrame a été modifié.

    get_actors_movies_dataframe(cleaned: bool = False, modify: bool = False) -> pd.DataFrame:
        Récupère le DataFrame des films d'acteurs, nettoyé ou non.

    get_directors_movies_dataframe(modify: bool = False, cleaned: bool = False) -> pd.DataFrame:
        Récupère le DataFrame des films de réalisateurs, nettoyé ou non.

    get_dataframes(name: str, cleaned: bool = False) -> pd.DataFrame:
        Récupère le DataFrame demandé, nettoyé ou non.
    """
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

    def load_dataframe(self, path: str, funcs: callable) -> pd.DataFrame:
        """
        Charge ou crée un DataFrame à partir d'un chemin spécifié.

        Vérifie l'existence d'un fichier au chemin donné. Si absent,
        utilise `funcs` pour créer un nouveau DataFrame. Sinon, charge le
        DataFrame existant.

        Parameters
        ----------
        path : str
            Chemin du fichier à charger.
        funcs : callable
            Fonction pour créer un DataFrame si absent.

        Returns
        -------
        pd.DataFrame
            DataFrame chargé ou créé.
        """


        name = path.split("/")[-1]
        if not os.path.exists(path):
            logging.info(f"File {name} not found. Creation...")
            return funcs()
        else:
            return import_datasets(path, "parquet")


    def get_cleaned_movies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie un DataFrame de films selon les critères de configuration.

        Filtre les films basé sur l'année de sortie, la note moyenne, le nombre
        de votes, et la durée. Les critères sont dans `config`.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame des films à nettoyer.

        Returns
        -------
        pd.DataFrame
            DataFrame nettoyé avec les films filtrés.
        """
        condi = (
            (df["titre_date_sortie"] >= self.config["movies_years"]) &
            (df["rating_avg"] >= self.config["movies_rating_avg"]) &
            (df["rating_votes"] >= self.config["movies_min_votes"]) &
            ~(
                (df["titre_duree"] < self.config["movies_min_duration"]) |
                (df["titre_duree"] > self.config["movies_max_duration"])
            )
        )
        df = df[condi].reset_index(drop=True)
        self.config["movies_min_votes"] = int(df["rating_votes"].min())
        hjson_dump(self.config)
        return df


    def update_movies(self, path_file: str) -> pd.DataFrame:
        """
        Met à jour et nettoie le DataFrame des films.

        Charge, nettoie, exclut certains genres, et sauvegarde le DataFrame
        des films dans le chemin spécifié.

        Parameters
        ----------
        path_file : str
            Chemin de sauvegarde du DataFrame nettoyé.

        Returns
        -------
        pd.DataFrame
            DataFrame des films mis à jour.
        """
        movies_path = f"{self.default_path}/movies.parquet"
        df = import_datasets(movies_path, "parquet")
        df = self.get_cleaned_movies(df)
        genres_ = ["Documentary", "Reality-TV", "News"]
        df = df[df['titre_genres'].apply(lambda x: all(g not in x for g in genres_))]
        df.to_parquet(path_file)
        return df


    def get_movies_dataframe(self, cleaned: bool = False) -> pd.DataFrame:
        """
        Récupère le DataFrame des films, nettoyé ou non.

        Charge le DataFrame des films depuis le chemin par défaut. Si `cleaned`
        est vrai, applique un nettoyage supplémentaire.

        Parameters
        ----------
        cleaned : bool, optional
            Indique si le nettoyage est requis (défaut False).

        Returns
        -------
        pd.DataFrame
            DataFrame des films, éventuellement nettoyé.
        """
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
                    "parquet"
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

                # col_list = ["spoken_languages", "production_countries"]
                # merged = eda.clean_square_brackets(
                #     merged,
                #     col_list
                # )
                # merged = merged.dropna()
                # logging.info(
                #     f"Length dataframe merged cleaned : {len(merged)}")

                # merged = eda.apply_decode_and_split(
                #     merged,
                #     col_list,
                #     decode_clean
                # )
                akas = import_datasets(
                    self.tsv_file["title_akas"],
                    types="pandas",
                    sep="\t"
                )

                akas = akas[akas["region"] == self.config["movies_region"].upper()]
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
                # a supprimer si API
                condi = (
                    df["status"] == "Released"
                    # df["status"] == self.config["movies_status"].title()
                )
                df = df[condi]
                ###################

                df.drop(
                    ["titleId"],
                    inplace=True,
                    axis=1
                )
                df = df.reset_index(drop="index")
                df = eda.split_columns(df, "titre_genres")
                df = eda.apply_decade_column(df)
                df = eda.drop_nan_values(df)
                logging.info("Replace OG title if latin...")
                df['titre_str'] = df.apply(eda.replace_title_if_latin, axis=1)
                df = df.reset_index(drop="index")
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
                    logging.info(f"Values modified ? {self.check_if_moded(df)}")
                    logging.info("Values modified, creating new cleaned movies...")
                    df = self.update_movies(path_file)
                else:
                    logging.info(f"Values modified ? {self.check_if_moded(df)}")
                    logging.info("No need to update movies, all values are equals.")
            else:
                df = self.update_movies(path_file)
            logging.info(f"Dataframe {name} ready to use!")
        return df

    def get_persons_dataframes(self) -> pd.DataFrame:
        """
        Récupère le DataFrame des personnes.

        Charge ou crée le DataFrame des personnes, contenant des informations
        sur diverses personnalités (acteurs, réalisateurs, etc.).

        Returns
        -------
        pd.DataFrame
            DataFrame contenant les données des personnes.
        """
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

    def get_characters_dataframe(self) -> pd.DataFrame:
        """
        Récupère le DataFrame des personnages.

        Charge ou crée le DataFrame des personnages de films, basé sur les
        informations de casting et de rôles.

        Returns
        -------
        pd.DataFrame
            DataFrame des personnages.
        """
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

    def get_actors_dataframe(self) -> pd.DataFrame:
        """
        Récupère le DataFrame des acteurs.

        Charge ou crée le DataFrame contenant les informations sur les acteurs,
        filtré à partir du DataFrame des personnages.

        Returns
        -------
        pd.DataFrame
            DataFrame des acteurs.
        """
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

    def get_directors_dataframe(self) -> pd.DataFrame:
        """
        Récupère le DataFrame des réalisateurs.

        Charge ou crée le DataFrame contenant les informations sur les réalisateurs,
        filtré à partir du DataFrame des personnages.

        Returns
        -------
        pd.DataFrame
            DataFrame des réalisateurs.
        """
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
        """
        Vérifie si le DataFrame a été modifié selon la configuration.

        Compare les paramètres de configuration avec les valeurs minimales
        dans le DataFrame pour déterminer si une mise à jour est nécessaire.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame à vérifier.

        Returns
        -------
        bool
            Renvoie True si une modification est nécessaire, sinon False.
        """
        check_year = int(df["titre_date_sortie"].min())
        check_min_duration = int(df["titre_duree"].min())
        check_max_duration = int(df["titre_duree"].max())
        check_rating = float(df["rating_avg"].min())
        check_votes = int(df["rating_votes"].min())
        return (
            check_year != self.config["movies_years"] or
            check_min_duration != self.config["movies_min_duration"] or
            check_max_duration != self.config["movies_max_duration"] or
            check_rating != self.config["movies_rating_avg"] or
            check_votes != self.config["movies_min_votes"]
        )

    def get_actors_movies_dataframe(
        self, cleaned: bool = False, modify: bool = False
    ) -> pd.DataFrame:
        """
        Récupère le DataFrame des films avec acteurs, nettoyé ou non.

        Charge ou crée le DataFrame associant les acteurs à leurs films.
        Applique un nettoyage et/ou des modifications si nécessaire.

        Parameters
        ----------
        cleaned : bool, optional
            Indique si un nettoyage est requis (défaut False).
        modify : bool, optional
            Indique si des modifications sont requises (défaut False).

        Returns
        -------
        pd.DataFrame
            DataFrame des films avec acteurs.
        """
        name = "actors_movies"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file) and not modify:
            movies_actors = import_datasets(
                path_file,
                "parquet"
            )
            if self.check_if_moded(movies_actors):
                logging.info("Updating...")
                return self.get_actors_movies_dataframe(
                    cleaned=cleaned, modify=True
                )
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
    ) -> pd.DataFrame:
        """
        Récupère le DataFrame des films avec réalisateurs.

        Charge ou crée le DataFrame associant les réalisateurs à leurs films.
        Applique un nettoyage et/ou des modifications si nécessaire.

        Parameters
        ----------
        modify : bool, optional
            Indique si des modifications sont requises (défaut False).
        cleaned : bool, optional
            Indique si un nettoyage est requis (défaut False).

        Returns
        -------
        pd.DataFrame
            DataFrame des films avec réalisateurs.
        """
        name = "directors_movies"
        path_file = f"{self.default_path}/{name}.parquet"
        if os.path.exists(path_file) and not modify:
            movies_directors = import_datasets(
                path_file,
                "parquet"
            )
            if self.check_if_moded(movies_directors):
                logging.info("Updating...")
                return self.get_directors_movies_dataframe(
                    cleaned=cleaned, modify=True
                )
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

    def get_machine_learning_dataframe(
        self, cleaned: bool = False, modify: bool = False
    ) -> pd.DataFrame:
        """
        Récupère le DataFrame pour l'apprentissage machine.

        Charge ou crée un DataFrame spécifique pour des besoins en apprentissage
        machine, avec des données nettoyées et structurées.

        Parameters
        ----------
        cleaned : bool, optional
            Nettoyage des données requis (défaut False).
        modify : bool, optional
            Modifications des données requises (défaut False).

        Returns
        -------
        pd.DataFrame
            DataFrame préparé pour l'apprentissage machine.
        """
        name = "machine_learning"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file) and not modify:
            ml_df = import_datasets(
                path_file,
                "parquet"
            )
            logging.info(f"Dataframe {name} ready to use!")
            return ml_df
        else:
            logging.info(f"Creating {name} dataframe...")
            tmdb_l = "clean_datasets/tmdb_updated.parquet"
            actors_l = "clean_datasets/actors_movies.parquet"
            directors_l = "clean_datasets/directors_movies.parquet"
            movies_l = "clean_datasets/movies_cleaned.parquet"

            tmdb = import_datasets(tmdb_l, "parquet")
            actors = import_datasets(actors_l, "parquet")
            directors = import_datasets(directors_l, "parquet")
            movies = import_datasets(movies_l, "parquet")

            col_to_keep = [
                "titre_id",
                "titre_str",
                "titre_genres",
                "rating_avg",
                "rating_votes"
            ]
            movies = movies[col_to_keep]

            col_to_keep = [
                "imdb_id",
                "overview",
                "popularity",
                "poster_path",
                "revenue",
                "release_date",
            ]
            tmdb = tmdb[col_to_keep]

            col_to_keep = [
                "titre_id",
                "person_name",
                # "person_index"
            ]
            actors = actors[col_to_keep]

            col_to_keep = [
                "titre_id",
                "person_name",
                # "person_index"
            ]
            directors = directors[col_to_keep]

            logging.info(f"Creating {name} dataframe...")
            directors_list_id = directors["titre_id"].unique()
            condi = movies["titre_id"].isin(directors_list_id)
            condi2 = actors["titre_id"].isin(directors_list_id)
            movies = movies[condi]
            actors = actors[condi2]

            actors_list_id = actors["titre_id"].unique()
            condi = movies["titre_id"].isin(actors_list_id)
            condi2 = directors["titre_id"].isin(actors_list_id)
            movies = movies[condi]
            directors = directors[condi2]


            actors.loc[:, "person_name"] = actors["person_name"].str.split(", ")
            directors.loc[:, "person_name"] = directors["person_name"].str.split(", ")

            person_name = actors.groupby("titre_id")["person_name"].sum().reset_index()
            person_list = person_name["person_name"].to_list()

            directors_name = directors.groupby("titre_id")["person_name"].sum().reset_index()
            directors_list = directors_name["person_name"].to_list()

            movies["actors"] = person_list
            movies["directors"] = directors_list


            logging.info(f"Merging {name} dataframe...")
            ml_df = pd.merge(
                movies,
                tmdb,
                left_on = "titre_id",
                right_on = "imdb_id"
            )

            logging.info(f"Droping NaN {name} dataframe...")
            ml_df.drop(["imdb_id"], axis = 1, inplace = True)
            ml_df[ml_df.isna().any(axis=1)]
            ml_df.dropna(inplace=True)
            ml_df.reset_index(drop="index", inplace=True)

            tt = [
                "actors",
                "titre_genres",
                "directors",
            ]
            for t in tt:
                ml_df[t] = ml_df[t].apply(
                    lambda x: ", ".join(map(str, x))
                ).replace(" ", "")

            # Full loWer pour reduire les titres, actors, directors etc...
            # for t in tt:
            #     ml_df[t[0]] = ml_df[t[1]].apply(full_lower)
            # logging.info(f"Process Overview...")
            # ml_df['overview'] = ml_df['overview'].astype(str).apply(clean_overview)

            logging.info(f"Writing {name} dataframe...")
            ml_df.to_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return ml_df

    def get_dataframes(
        self, name: str, cleaned: bool = False
    ) -> pd.DataFrame:
        """
        Récupère un DataFrame spécifique par son nom.

        Selon le nom donné, cette méthode charge ou crée le DataFrame
        correspondant, avec une option de nettoyage.

        Parameters
        ----------
        name : str
            Nom du DataFrame à récupérer.
        cleaned : bool, optional
            Indique si un nettoyage est requis (défaut False).

        Returns
        -------
        pd.DataFrame
            DataFrame demandé.
        """
        downloader(self.config)
        if name.lower() == "movies":
            return self.get_movies_dataframe()
        elif name.lower() == "movies_cleaned":
            return self.get_movies_dataframe(cleaned=cleaned)
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
        elif name.lower() == "machine_learning":
            return self.get_machine_learning_dataframe(cleaned=cleaned)
        else:
            raise KeyError(f"{name.capitalize()} not know!")


    def get_all_dataframes(self):
        """
        Récupère tous les DataFrames principaux.

        Cette méthode parcourt une liste prédéfinie de noms de DataFrames
        et les charge ou les crée un par un. Elle utilise `get_dataframes`
        pour chaque type de DataFrame. Les opérations et leur achèvement
        sont enregistrés dans les logs.

        Chaque DataFrame est identifié par son nom et une couleur associée
        pour le logging. La méthode assure la création de DataFrames pour
        les films, films nettoyés, acteurs, réalisateurs, et données pour
        l'apprentissage machine.
        """
        names = (
            ("movies", "#efc3a4"),
            ("movies_cleaned", "#cfe2f3"),
            ("actors_movies", "#ffd47b"),
            ("directors_movies", "#6fa8dc"),
            ("machine_learning", "#94e5df"),
        )
        for name in names:
            txt = color("-"*10 + f" Start creating {name[0]} " + "-"*10, color=name[1])
            logging.info(txt)
            self.get_dataframes(name[0], True)
            txt = color("-"*10 + f" Job Done for {name[0]} ! " + "-"*10 + "\n", color=name[1])
            logging.info(txt)

        txt = color("-"*20 + f" Job Done for {len(names)} dataframes ! " + "-"*20 + "\n", color="green")
        logging.info(txt)
