import polars as pl
import ast

from datetime import datetime

def decode_clean(item: str) -> str:
    return item.replace("[", "").replace("]", "").replace("'", "").replace('"', "")

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s) if isinstance(s, str) else s
    except:
        return []

def order_and_rename(
    df: pl.DataFrame, og_col: list, new_col_name: list
) -> pl.DataFrame:
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
        [pl.col(old).alias(new) for old, new in zip(og_col, new_col_name)]
    )


def col_to_keep(datasets: str) -> list:
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
            "original_language",
            "original_title",
            "popularity",
            "production_countries",
            "revenue",
            "spoken_languages",
            "status",
            "region",
            "cuts",
        ]
    if datasets in ["actors", "directors"]:
        return [
            "nconst",
            "primaryName",
            "birthYear",
            "category",
            "knownForTitles",
            "ordering",
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
            "nconst",
            "primaryName",
            "birthYear",
            "category",
            "characters",
            "knownForTitles",
            "ordering",
        ]
    else:
        raise KeyError(f"{datasets} n'est pas valide!")


def col_renaming(datasets: str) -> list:
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
            "original_language",
            "original_title",
            "popularity",
            "production_countries",
            "revenue",
            "spoken_languages",
            "status",
            "region",
            "cuts",
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


def apply_decade_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ajoute une colonne 'cuts' au DataFrame Polars qui catégorise les années de sortie
    des titres en décennies.
    """
    year = datetime.now().year
    bins, names = bins_generator(year)

    def map_to_decade(year):
        for i, bin in enumerate(bins):
            if year < bin:
                return names[i-1] if i > 0 else f"before {bins[0]}"
        return f"{bins[-1]}+"

    df = df.with_columns(pl.col("startYear").apply(map_to_decade).alias("cuts"))
    return df
