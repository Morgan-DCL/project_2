import os

import pandas as pd
import polars as pl

pd.set_option("display.float_format", lambda x: f"{x :.2f}")
from cleaner import DataCleaner
from downloader import downloader
from polars_tools import (
    apply_decade_column_pl,
    col_renaming_pl,
    col_to_keep_pl,
    import_datasets_pl,
    order_and_rename_pl,
)
from tools import (
    clean_overview,
    col_renaming,
    col_to_keep,
    color,
    full_lower,
    get_tsv_files,
    hjson_dump,
    if_tt_remove,
    import_datasets,
    logging,
    make_filepath,
    order_and_rename,
    replace_ids_with_titles,
    supprimer_accents,
)

clean = DataCleaner()


class GetDataframes:
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

    def __init__(self, config: dict):
        self.config = config
        self.default_path = make_filepath(config["clean_df_path"])
        self.download_path = make_filepath(config["download_path"])
        self.tsv_file = get_tsv_files(self.download_path)
        self.fix_n = "\\N"

    def load_dataframe(self, path: str, funcs: callable):
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
            return import_datasets_pl(path, "parquet")

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
            (df["titre_date_sortie"] >= self.config["movies_years"])
            & (df["rating_avg"] >= self.config["movies_rating_avg"])
            & (df["rating_votes"] >= self.config["movies_min_votes"])
            & ~(
                (df["titre_duree"] < self.config["movies_min_duration"])
                | (df["titre_duree"] > self.config["movies_max_duration"])
            )
        )
        df = df[condi].reset_index(drop=True)
        self.config["movies_min_votes"] = int(df["rating_votes"].min())
        hjson_dump(self.config)
        return df

    def update_movies(self, path_file: str) -> pl.DataFrame:
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
        df = df[
            df["titre_genres"].apply(
                lambda x: all(g not in x for g in genres_)
            )
        ]
        df.to_parquet(path_file)
        df = pl.from_pandas(df)
        return df

    def get_movies_dataframe(self, cleaned: bool = False) -> pl.DataFrame:
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
                df = import_datasets_pl(path_file, "parquet")
            else:
                first_df = import_datasets_pl(
                    self.tsv_file["title_basics"],
                    "polars",
                    sep="\t",
                    fix=self.fix_n,
                )
                logging.info(f"Filter Porn Movies...!")
                movies = first_df.filter(
                    (pl.col("titleType") == "movie")
                    & (pl.col("isAdult") == 0)
                )
                title_ratings = import_datasets_pl(
                    self.tsv_file["title_ratings"],
                    "polars",
                    sep="\t",
                    fix=self.fix_n,
                )
                logging.info(f"Join !")
                mov_rating = movies.join(
                    title_ratings,
                    left_on="tconst",
                    right_on="tconst",
                )
                df_imdb = import_datasets_pl(
                    self.tsv_file["imdb_full"], "parquet"
                )
                merged = mov_rating.join(
                    df_imdb,
                    left_on="tconst",
                    right_on="imdb_id",
                )
                merged = merged.drop(clean.columns_to_drop_tmdb())
                merged = merged.rename({"genres_right": "genres"})

                akas = import_datasets_pl(
                    self.tsv_file["title_akas"],
                    "polars",
                    sep="\t",
                    fix=self.fix_n,
                )

                region_only = akas.select(["titleId", "region"])
                fr_only = region_only.filter(pl.col("region") == "FR")
                logging.info(f"Join !")
                df = merged.join(
                    fr_only,
                    left_on="tconst",
                    right_on="titleId",
                )
                filtered = df.filter(pl.col("status") == "Released")
                logging.info(f"Apply Decade !")
                filtered = apply_decade_column_pl(filtered)

                ordered = order_and_rename_pl(
                    filtered,
                    col_to_keep_pl("movies"),
                    col_renaming_pl("movies"),
                )
                drop_nan = ordered.drop_nulls()
                df = drop_nan.filter(~pl.col("titre_id").is_duplicated())
                logging.info(f"drop_nan = {len(ordered) - len(drop_nan)}")
                logging.info(f"drop_dup = {len(drop_nan) - len(df)}")

                df.write_parquet(path_file)
            logging.info(f"Dataframe {name} ready to use!")
        else:
            name = "movies_cleaned"
            path_file = f"{self.default_path}/{name}.parquet"
            if os.path.exists(path_file):
                df = import_datasets_pl(path_file, "parquet")
                if self.check_if_moded(df):
                    logging.info(
                        f"Values modified ? {self.check_if_moded(df)}"
                    )
                    logging.info(
                        "Values modified, creating new cleaned movies..."
                    )
                    df = self.update_movies(path_file)
                else:
                    logging.info(
                        f"Values modified ? {self.check_if_moded(df)}"
                    )
                    logging.info(
                        "No need to update movies, all values are equals."
                    )
            else:
                df = self.update_movies(path_file)
            logging.info(f"Dataframe {name} ready to use!")
        return df

    def get_persons_dataframes(self) -> pl.DataFrame:
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
            df = import_datasets_pl(path_file, "parquet")
        else:
            # logging.info("Import Name Basics !")
            first_df = import_datasets_pl(
                self.tsv_file["name_basics"],
                "polars",
                sep="\t",
                fix=self.fix_n,
            )
            first_df = first_df.drop(["deathYear", "primaryProfession"])
            first_df = first_df.fill_null("Unknow")
            first_df = first_df.fill_null(0)

            logging.info("Spliting and modifing dtypes...")
            df = first_df.with_columns(
                pl.when(pl.col("knownForTitles").is_not_null())
                .then(pl.col("knownForTitles").str.split(","))
                .alias("person_movies")
            ).drop("knownForTitles")

            logging.info(f"Writing {name} dataframe...")
            df.write_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return df

    def get_characters_dataframe(self) -> pl.DataFrame:
        """
        Récupère le DataFrame des personnages.

        Charge ou crée le DataFrame des personnages de films, basé sur les
        informations de casting et de rôles.

        Returns
        -------
        pd.DataFrame
            DataFrame des personnages.
        """
        name = "characters"
        path_file = f"{self.default_path}/{name}.parquet"

        if os.path.exists(path_file):
            df = import_datasets_pl(path_file, "parquet")
        else:
            first_df = import_datasets_pl(
                self.tsv_file["title_principals"],
                "polars",
                sep="\t",
                fix=self.fix_n,
            )
            first_df = first_df.drop(["job", "characters", "ordering"])
            logging.info("Spliting and modifing dtypes...")
            first_df = first_df.fill_null("Unknow")
            df = first_df.with_columns(
                pl.when(pl.col("category").is_not_null())
                .then(pl.col("category").str.split(","))
                .alias("category")
            )
            logging.info(f"Writing {name} dataframe...")
            df.write_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return df

    def get_actors_dataframe(self) -> pl.DataFrame:
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
            df = import_datasets_pl(path_file, "parquet")
        else:
            df = self.load_dataframe(
                f"{self.default_path}/characters.parquet",
                self.get_characters_dataframe,
            )
            logging.info(f"Get {name} only...")
            actors_list = ["self", "actor", "actress"]
            df = (
                df.explode("category")
                .filter(pl.col("category").is_in(actors_list))
                .with_columns(pl.col("category").str.split(","))
            )
            logging.info(f"Writing {name} dataframe...")
            df.write_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return df

    def get_directors_dataframe(self) -> pl.DataFrame:
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
            df = import_datasets_pl(path_file, "parquet")
        else:
            df = self.load_dataframe(
                f"{self.default_path}/characters.parquet",
                self.get_characters_dataframe,
            )
            logging.info(f"Get {name} only...")
            df = (
                df.explode("category")
                .filter(pl.col("category").str.contains("director"))
                .with_columns(pl.col("category").str.split(","))
            )
            logging.info(f"Writing {name} dataframe...")
            df.write_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return df

    def check_if_moded(self, df: pl.DataFrame) -> bool:
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
        df = df.to_pandas()
        check_year = int(df["titre_date_sortie"].min())
        check_min_duration = int(df["titre_duree"].min())
        check_max_duration = int(df["titre_duree"].max())
        check_rating = float(df["rating_avg"].min())
        check_votes = int(df["rating_votes"].min())
        return (
            check_year != self.config["movies_years"]
            or check_min_duration != self.config["movies_min_duration"]
            or check_max_duration != self.config["movies_max_duration"]
            or check_rating != self.config["movies_rating_avg"]
            or check_votes != self.config["movies_min_votes"]
        )

    def get_actors_movies_dataframe(
        self, cleaned: bool = False, modify: bool = False
    ) -> pl.DataFrame:
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
            movies_actors = import_datasets_pl(path_file, "parquet")
            if self.check_if_moded(movies_actors):
                logging.info("Updating...")
                return self.get_actors_movies_dataframe(
                    cleaned=cleaned, modify=True
                )
            else:
                logging.info(f"Dataframe {name} ready to use!")
                return movies_actors
        else:
            actors = self.load_dataframe(
                f"{self.default_path}/actors.parquet",
                self.get_actors_dataframe,
            )
            persons = self.load_dataframe(
                f"{self.default_path}/persons.parquet",
                self.get_persons_dataframes,
            )
            if cleaned:
                movies = self.load_dataframe(
                    f"{self.default_path}/movies_cleaned.parquet",
                    self.get_movies_dataframe(True),
                )
            else:
                movies = self.load_dataframe(
                    f"{self.default_path}/movies.parquet",
                    self.get_movies_dataframe,
                )
            logging.info("Join actors with persons!")
            actors_names = actors.join(persons, on="nconst")
            logging.info("Join actors with movies!")
            movies_actors = movies.join(
                actors_names, left_on="titre_id", right_on="tconst"
            )

            movies_actors = order_and_rename_pl(
                movies_actors,
                col_to_keep_pl("actors_movies"),
                col_renaming_pl("actors_movies"),
            )
            movies_actors = movies_actors.to_pandas()

            logging.info("Replace tt by movies titles...")
            dict_titre = (
                movies_actors[["titre_id", "titre_str"]]
                .drop_duplicates()
                .set_index("titre_id")
                .to_dict()["titre_str"]
            )
            movies_actors["person_film"] = movies_actors[
                "person_film"
            ].apply(
                lambda x: if_tt_remove(
                    replace_ids_with_titles(x, dict_titre)
                )
            )
            movies_actors = pl.from_pandas(movies_actors)
            logging.info(f"Writing {name} dataframe...")
            movies_actors.write_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return movies_actors

    def get_directors_movies_dataframe(
        self, modify: bool = False, cleaned: bool = False
    ) -> pl.DataFrame:
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
            movies_directors = import_datasets_pl(path_file, "parquet")
            if self.check_if_moded(movies_directors):
                logging.info("Updating...")
                return self.get_directors_movies_dataframe(
                    cleaned=cleaned, modify=True
                )
            else:
                logging.info(f"Dataframe {name} ready to use!")
                return movies_directors
        else:
            directors = self.load_dataframe(
                f"{self.default_path}/directors.parquet",
                self.get_directors_dataframe,
            )
            persons = self.load_dataframe(
                f"{self.default_path}/persons.parquet",
                self.get_persons_dataframes,
            )
            if cleaned:
                movies = self.load_dataframe(
                    f"{self.default_path}/movies_cleaned.parquet",
                    self.get_movies_dataframe(True),
                )
            else:
                movies = self.load_dataframe(
                    f"{self.default_path}/movies.parquet",
                    self.get_movies_dataframe,
                )

            directors_names = directors.join(persons, on="nconst")

            movies_directors = movies.join(
                directors_names,
                left_on="titre_id",
                right_on="tconst",
            )

            movies_directors = order_and_rename_pl(
                movies_directors,
                col_to_keep_pl("directors_movies"),
                col_renaming_pl("directors_movies"),
            )
            movies_directors = movies_directors.to_pandas()

            logging.info("Replace tt by movies titles...")
            dict_titre = (
                movies_directors[["titre_id", "titre_str"]]
                .drop_duplicates()
                .set_index("titre_id")
                .to_dict()["titre_str"]
            )
            movies_directors["person_film"] = movies_directors[
                "person_film"
            ].apply(
                lambda x: if_tt_remove(
                    replace_ids_with_titles(x, dict_titre)
                )
            )
            logging.info(f"Writing {name} dataframe...")
            movies_directors = pl.from_pandas(movies_directors)
            movies_directors.write_parquet(path_file)
        logging.info(f"Dataframe {name} ready to use!")
        return movies_directors

    def get_machine_learning_dataframe(
        self, cleaned: bool = False, modify: bool = False
    ) -> pd.DataFrame:
        name = "machine_learning"
        path_file = f"{self.default_path}/{name}.parquet"
        name_og = "machine_learning_final"
        path_final_file = f"{self.default_path}/{name_og}.parquet"

        if os.path.exists(path_final_file) and not modify:
            ml_df = import_datasets(path_final_file, "parquet")
            logging.info(f"Dataframe {name_og} ready to use!")
            return ml_df
        else:
            first_df = import_datasets(path_file, "parquet")
            ml_df = order_and_rename(
                first_df, col_to_keep(name), col_renaming(name)
            )
            to_clean = [
                "actors",
                "titre_genres",
                "director",
                "keywords",
            ]
            for t in to_clean:
                ml_df[t] = (
                    ml_df[t]
                    .apply(lambda x: ", ".join(map(str, x)))
                    .replace(" ", "")
                )
            ml_df["titre_clean"] = ml_df["titre_str"]
            ml_df["titre_clean"] = ml_df["titre_clean"].apply(
                lambda x: x.lower()
            )
            ml_df["date"] = pd.to_datetime(ml_df["date"])
            ml_df["date"] = ml_df["date"].dt.year
            ml_df.reset_index(drop="index", inplace=True)
            ml_df.to_parquet("clean_datasets/site_web.parquet")
            logging.info("Cleaning StopWords and Lemmatize...")
            to_clean.extend(["titre_clean", "overview"])
            for col in to_clean:
                ml_df[col] = (
                    ml_df[col].astype(str).apply(supprimer_accents)
                )
            ml_df["overview"] = (
                ml_df["overview"].astype(str).apply(clean_overview)
            )

            to_clean.remove("titre_clean")
            for t in to_clean:
                logging.info(f"lowering everything in {t}")
                ml_df[t] = ml_df[t].apply(full_lower)
            ml_df = ml_df[col_renaming(name)]
            ml_df.reset_index(drop="index", inplace=True)
            ml_df.to_parquet(path_final_file)
        logging.info(f"Dataframe machine_learning_final ready to use!")
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
            return self.get_movies_dataframe().to_pandas()
        elif name.lower() == "movies_cleaned":
            return self.get_movies_dataframe(cleaned=cleaned).to_pandas()
        elif name.lower() == "persons":
            return self.get_persons_dataframes().to_pandas()
        elif name.lower() == "characters":
            return self.get_characters_dataframe().to_pandas()
        elif name.lower() == "actors":
            return self.get_actors_dataframe().to_pandas()
        elif name.lower() == "directors":
            return self.get_directors_dataframe().to_pandas()
        elif name.lower() == "actors_movies":
            return self.get_actors_movies_dataframe(
                cleaned=cleaned
            ).to_pandas()
        elif name.lower() == "directors_movies":
            return self.get_directors_movies_dataframe(
                cleaned=cleaned
            ).to_pandas()
        elif name.lower() == "machine_learning":
            return self.get_machine_learning_dataframe()
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
            # ("movies", "#efc3a4"),
            # ("movies_cleaned", "#cfe2f3"),
            # ("actors_movies", "#ffd47b"),
            # ("directors_movies", "#6fa8dc"),
            # ("machine_learning", "#94e5df"),
            ("movies", "green"),
            ("movies_cleaned", "green"),
            ("actors_movies", "green"),
            ("directors_movies", "green"),
            ("machine_learning", "green"),
        )
        for name in names:
            txt = color(
                "-" * 10 + f" Start creating {name[0]} " + "-" * 10,
                color=name[1],
            )
            logging.info(txt)
            self.get_dataframes(name[0], True)
            txt = color(
                "-" * 10 + f" Job Done for {name[0]} ! " + "-" * 10 + "\n",
                color=name[1],
            )
            logging.info(txt)

        txt = color(
            "-" * 20
            + f" Job Done for {len(names)} dataframes ! "
            + "-" * 20
            + "\n",
            color="green",
        )
        logging.info(txt)
