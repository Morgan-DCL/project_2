import logging

import pandas as pd
import numpy as np
import pandas as pd

pd.set_option("display.float_format", lambda x: f"{x :.2f}")
from datetime import datetime

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
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
    return df.replace("\\N", 0, inplace=True)


def fix_Neat(
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
    return df.replace("\\N", 0)


def fix_encode_(column: str) -> str:
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
    if isinstance(column, str):
        return column.encode("latin1").decode("utf-8")
    return column


def fix_encode_df(df: pd.DataFrame) -> pd.DataFrame:
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
        df[col] = df[col].apply(fix_encode_)
    return df


class DataCleaner:
    def __init__(self):
        pass

    def fix_values(
        self,
        datas: pd.DataFrame,
        method: str = "fix_n",
    ):
        """
        Répare les valeurs dans un DataFrame en utilisant une méthode spécifique.

        Parameters
        ----------
        datas : pd.DataFrame
            Le DataFrame contenant les données à réparer.
        method : str, optional
            La méthode à utiliser pour la réparation. Les options sont "fix_n",
            "fix_neat" et "fix_encode". Par défaut, "fix_n" est utilisé.

        Returns
        -------
        pd.DataFrame
            Le DataFrame avec les valeurs réparées.

        Raises
        ------
        ValueError
            Si la méthode spécifiée n'est pas reconnue.
        """

        if method == "fix_n":
            logging.info("Fixing N values...")
            return datas.apply(fix_N)
        if method == "fix_neat":
            logging.info("Fixing N values...")
            return datas.apply(fix_Neat)
        elif method == "fix_encode":
            logging.info("Fixing encoding values...")
            return fix_encode_df(datas)
        else:
            raise ValueError(f"{method} not recognized!")

    def clean_porn(
        self, datas: pd.DataFrame, columns_name: str = "titre_genres"
    ):
        """
        Nettoie les films pornographiques du DataFrame fourni.

        Cette fonction supprime les lignes contenant le mot 'Adult' dans la colonne spécifiée.
        Elle utilise la méthode 'str.contains' pour identifier ces lignes et les supprime du DataFrame.

        Parameters
        ----------
        datas : pd.DataFrame
            Le DataFrame à nettoyer.
        columns_name : str, optional
            Le nom de la colonne à vérifier pour le mot 'Adult'. Par défaut, c'est "titre_genres".

        Returns
        -------
        pd.DataFrame
            Le DataFrame nettoyé, sans les lignes contenant le mot 'Adult' dans la colonne spécifiée.
        """
        logging.info("Cleaning porn movies...")
        datas = datas[datas[columns_name] != 0]
        msk = datas[columns_name].str.contains("Adult")
        return datas[~msk]

    def split_columns(self, df: pd.DataFrame, columns: str) -> pd.DataFrame:
        """
        Sépare les éléments de la colonne spécifiée dans un DataFrame, où les éléments
        sont des chaînes de caractères séparées par des virgules, en listes.

        Parameters
        ----------
        df : pd.DataFrame
            Le DataFrame contenant la colonne à séparer.
        columns : str
            Le nom de la colonne dont les valeurs chaînes de caractères doivent être
            séparées en listes.

        Returns
        -------
        pd.DataFrame
            Le DataFrame avec la colonne spécifiée maintenant contenant des listes
            d'éléments au lieu de chaînes de caractères séparées par des virgules.

        """
        logging.info(f"{columns.capitalize()} splited !")
        df[columns] = df[columns].str.split(",")
        return df

    def get_top_genres(sef, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renvoie le genre le plus fréquent dans la colonne 'titre_genres' d'un DataFrame.

        Cette fonction prend un DataFrame qui contient une colonne 'titre_genres',
        où chaque entrée peut être une liste de genres. Elle utilise la méthode
        `explode` pour séparer les genres, puis trouve le mode, c'est-à-dire le
        genre qui apparaît le plus souvent.

        Parameters
        ----------
        df : pd.DataFrame
            Le DataFrame contenant la colonne 'titre_genres' avec des listes de genres.

        Returns
        -------
        pd.DataFrame
            Un DataFrame contenant le genre le plus fréquent.
        """
        genre = df["titre_genres"].explode()
        return genre.mode()[0]

    def apply_decade_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute une colonne 'cuts' au DataFrame qui catégorise les années de sortie
        des titres en décennies.

        La fonction génère des intervalles de décennies en fonction de l'année
        actuelle et utilise ces intervalles pour catégoriser chaque titre selon
        sa date de sortie.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contenant une colonne 'titre_date_sortie' avec des dates.

        Returns
        -------
        pd.DataFrame
            Le DataFrame original avec une nouvelle colonne 'cuts' contenant les
            étiquettes de décennie correspondantes aux années de sortie.
        """
        year = datetime.now().year
        bins, names = self.bins_generator(year)
        df["cuts"] = pd.cut(df["titre_date_sortie"], bins=bins, labels=names)
        logging.info(f"{len(bins)} cuts created, from {bins[1]} to {bins[-1]}")
        return df

    def columns_to_drop_tmdb(self):
        """
        Identifie les colonnes à supprimer dans un DataFrame TMDB.

        Cette fonction liste les colonnes qui ne sont pas nécessaires pour une
        analyse ultérieure des données TMDB (The Movie Database).

        Returns
        -------
        list of str
            Liste des noms de colonnes à exclure du DataFrame.
        """
        # status
        # popularity
        # revenue
        return [
            "adult",
            "backdrop_path",
            "budget",
            "genres",
            "homepage",
            "id",
            "imdb_id",
            # "original_title",
            "overview",
            "poster_path",
            "release_date",  # a voir si on arrive avec l'API pour ne garder que les films year + 1
            "runtime",
            "tagline",
            "title",
            "video",
            "vote_average",
            "vote_count",
            "production_companies_name",
            "production_companies_country",
        ]

    def drop_nan_values(self, df_og: pd.DataFrame) -> pd.DataFrame:
        """
        Supprime les lignes contenant des valeurs NaN dans un DataFrame.

        Cette fonction élimine toutes les lignes du DataFrame qui contiennent au moins
        une valeur NaN, puis enregistre dans le journal le nombre de lignes supprimées.

        Parameters
        ----------
        df_og : pd.DataFrame
            Le DataFrame original à nettoyer.

        Returns
        -------
        pd.DataFrame
            Un nouveau DataFrame sans les lignes contenant des valeurs NaN.
        """
        df = df_og.dropna()
        logging.info(f"Cleaned : {len(df_og) - len(df)} rows")
        return df

    def clean_square_brackets(
        self, df: pd.DataFrame, columns: list
    ) -> pd.DataFrame:
        """
        Nettoie les colonnes spécifiées d'un DataFrame en remplaçant les chaînes de
        caractères "[]" par des valeurs NaN.

        Parameters
        ----------
        df : pd.DataFrame
            Le DataFrame sur lequel effectuer le nettoyage.
        columns : list
            Une liste des noms des colonnes à nettoyer.

        Returns
        -------
        pd.DataFrame
            Le DataFrame avec les valeurs "[]" remplacées par NaN dans les colonnes
            spécifiées.
        """
        for col in columns:
            df[col] = np.where(df[col] == "[]", np.nan, df[col])
        return df

    def apply_decode_and_split(
        self, df: pd.DataFrame, columns: list, decode_func: callable
    ) -> pd.DataFrame:
        """
        Applique une fonction de décodage et sépare les chaînes de caractères
        dans les colonnes spécifiées d'un DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Le DataFrame sur lequel appliquer la transformation.
        columns : list
            Liste des noms de colonnes à traiter dans le DataFrame.
        decode_func : callable
            Fonction à appliquer à chaque élément des colonnes spécifiées.
            Doit retourner une chaîne de caractères qui sera ensuite séparée
            par des virgules.

        Returns
        -------
        pd.DataFrame
            Le DataFrame avec les colonnes spécifiées décodées et les chaînes
            de caractères séparées en listes.
        """
        for col in columns:
            df[col] = df[col].apply(decode_func).str.split(",")
        return df

    def is_latin(self, s: str) -> bool:
        """
        Vérifie si une chaîne de caractères est écrite en alphabet latin.

        Cette fonction tente d'encoder la chaîne `s` en utilisant l'encodage 'latin-1'.
        Si l'encodage réussit, cela indique que la chaîne est en alphabet latin.

        Parameters
        ----------
        s : str
            La chaîne de caractères à vérifier.

        Returns
        -------
        bool
            Renvoie True si `s` est en alphabet latin, False sinon.
        """
        try:
            s.encode("latin-1")
        except UnicodeEncodeError:
            return False
        else:
            return True

    def replace_title_if_latin(self, r: pd.Series) -> str:
        """
        Remplace le titre d'une série si certaines conditions sont remplies.

        Cette fonction remplace la valeur de 'titre_str' par
        celle de 'original_title'
        dans une ligne de DataFrame si la langue originale de
        cette ligne est 'fr' (français)
        et si 'original_title' est écrit en alphabet latin.

        Parameters
        ----------
        r : pd.Series
            Une ligne du DataFrame représentant un film ou une série,
            contenant les champs
            'original_language',
            'original_title', et 'titre_str'.

        Returns
        -------
        str
            Le titre modifié si les conditions sont remplies, sinon le 'titre_str' original.
        """
        if r["original_language"] == "fr" and self.is_latin(
            r["original_title"]
        ):
            return r["original_title"]
        else:
            return r["titre_str"]

    def bins_generator(self, max_date_df: int) -> tuple:
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
