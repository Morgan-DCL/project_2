import pandas as pd
pd.set_option('display.float_format', lambda x: f'{x :.2f}')
from cleaner import DataCleaner
import matplotlib.pyplot as plt
from datetime import datetime
from tools import (
    logging,
    bins_generator,
    import_datasets
)

def eda_movies(
    link: str,
) -> pd.DataFrame:
    """
    Importer la base de données movies et la clean.
    Nettoyage des N, on fixe l'encodage et modifie le type de données pour la colonne titre_date_sortie.
    """
    df_og = import_datasets(link, types="pandas", sep=",")
    cleaning = DataCleaner()
    df1 = cleaning.fix_values(df_og, "fix_n")
    df = cleaning.fix_values(df1, "fix_encode")
    logging.info(f"Cleaned : {len(df_og) - len(df)} rows")
    df['titre_date_sortie'].fillna(0, inplace=True)
    df['titre_date_sortie'] = df['titre_date_sortie'].astype("int64")
    df['titre_duree'] = df['titre_duree'].astype("int64")
    return df


def split_columns(
    df: pd.DataFrame,
    columns: str,
) -> pd.DataFrame:
    logging.info(f"{columns.capitalize()} splited !")
    df[columns] = df[columns].str.split(",")
    return df


def get_top_genres(
    df: pd.DataFrame,
) -> pd.DataFrame:
    genre = df["titre_genres"].explode()
    return genre.mode()[0]


def apply_decade_column(
    df: pd.DataFrame,
) -> pd.DataFrame:
    year = datetime.now().year
    bins, names = bins_generator(year)
    df["cuts"] = pd.cut(
        df["titre_date_sortie"],
        bins=bins,
        labels=names
    )
    logging.info(f"{len(bins)} cuts created, from {bins[1]} to {bins[-1]}")
    return df


def drop_nan_values(
    df_og: pd.DataFrame
) -> pd.DataFrame:
    df = df_og.dropna()
    logging.info(f"Cleaned : {len(df_og) - len(df)} rows")
    return df


def show_total_films_decade(
    df: pd.DataFrame
):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.hist(
        df['rating_avg'],
        bins=20,
        color='royalblue',
        edgecolor='black'
    )
    plt.axvline(
        x=df["rating_avg"].median(),
        color="red",
        linestyle="--",
        label="Mediane"
    )
    plt.title('Distribution des Notes Moyennes')
    plt.legend(loc='upper left')
    plt.xlabel('Note Moyenne')
    plt.ylabel('Fréquence')

    plt.subplot(1, 2, 2)
    total_films = df.groupby(
        "cuts",
        observed=True
    ).size().reset_index(name="total_films")

    x = total_films["cuts"]
    y = total_films["total_films"]
    bars = plt.bar(
        x,
        y,
        color='royalblue',
        edgecolor="black"
    )
    plt.title('Total des Films par Décénnie')
    plt.ylabel('Quantité de Film Produit')
    plt.xlabel('Année')
    plt.axhline(
        y=total_films["total_films"].median(),
        color="red",
        linestyle="--",
        label="Mediane",
        zorder=0
    )
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()