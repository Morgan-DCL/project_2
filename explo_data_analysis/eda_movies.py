import numpy as np
import pandas as pd
import ast

pd.set_option('display.float_format', lambda x: f'{x :.2f}')
from datetime import datetime

import matplotlib.gridspec as grid
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tools import bins_generator, logging


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
    logging.info(
        f"{len(bins)} cuts created, from {bins[1]} to {bins[-1]}"
    )
    return df

def columns_to_drop_tmdb():
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
        "release_date", # a voir si on arrive avec l'API pour ne garder que les films year + 1
        "runtime",
        "tagline",
        "title",
        "video",
        "vote_average",
        "vote_count",
        "production_companies_name",
        "production_companies_country"
    ]


def drop_nan_values(
    df_og: pd.DataFrame
) -> pd.DataFrame:
    df = df_og.dropna()
    logging.info(f"Cleaned : {len(df_og) - len(df)} rows")
    return df


def clean_square_brackets(
    df: pd.DataFrame,
    columns: list
) -> pd.DataFrame:
    for col in columns:
        df[col] = np.where(df[col] == "[]", np.nan, df[col])
    return df

def apply_decode_and_split(
    df: pd.DataFrame,
    columns: list,
    decode_func
):
    for col in columns:
        df[col] = df[col].apply(decode_func).str.split(",")
    return df


def show_total_films_decade(
    df: pd.DataFrame
):
    """
    Affiche le nombre total de films par décennie,
    la distribution des notes moyennes et le total des votes par décennie.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données sur les films.
        Il doit contenir les colonnes 'rating_avg', 'cuts' et 'rating_votes'.

    Notes
    -----
    Cette fonction utilise matplotlib pour générer trois graphiques :
    1. Un histogramme montrant la distribution des notes moyennes des films.
    2. Un graphique à barres montrant le nombre total de films produits par décennie.
    3. Un graphique à barres montrant le nombre total de votes reçus par décennie.

    Chaque graphique inclut une ligne indiquant la médiane de la distribution.

    La fonction ne renvoie rien mais affiche les graphiques à l'écran.
    """
    plt.figure(figsize=(16, 12))
    gs = grid.GridSpec(
        2, 2,
        width_ratios=[1, 1], height_ratios=[1, 1])

    plt.subplot(gs[0])
    # plt.subplot(1, 3, 1)
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
    plt.legend(loc='upper right')
    plt.xlabel('Note Moyenne')
    plt.ylabel('Fréquence')

    plt.subplot(gs[1])
    # plt.subplot(1, 3, 2)
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
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)

    plt.subplot(gs[2:4])
    # plt.subplot(1, 3, 3)
    rating_votes = df.groupby(
        "cuts",
        observed=True
    )["rating_votes"].sum().reset_index(name="vote_avg")

    x = rating_votes["cuts"]
    y = rating_votes["vote_avg"]
    bars = plt.bar(
        x,
        y,
        color='royalblue',
        edgecolor="black"
    )
    plt.title('Total des Votes par Décénnie')
    plt.ylabel('Quantité de Votes')
    plt.xlabel('Année')

    color = [
        ("#008080", "0.75"),
        ("#ff0000", "0.5"),
        ("#ffa500", "0.25")
    ]

    q = np.arange(0.25, 1, 0.25)

    quantile = df["rating_votes"].quantile(q).values
    for v, c in zip(quantile[::-1], color):
        plt.axhline(
            y=v,
            color=c[0],
            linestyle="--",
            label=c[1],
            zorder=0
        )
        offset = 200
        # plt.text(-0.7, v+offset, v, color=c[0])
        plt.annotate(
            str(v),
            xy=(-0.7, v),
            xycoords='data',
            textcoords='offset points',
            # arrowprops=dict(arrowstyle="->"),
            xytext=(0,10),  # positionnement du texte par rapport au point (x, y)
            color=c[0]
        )


    plt.legend(loc='upper right')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_total_films_decade_plotly(df: pd.DataFrame):
    # Histogramme des notes moyennes
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(
        x=df['rating_avg'],
        marker=dict(
            color='royalblue',
            line=dict(
                color='black', width=1)
        ),
        # name='Notes Moyennes',
        showlegend=False
    ))
    median = df["rating_avg"].median()
    max_ = df["rating_avg"].value_counts().max()
    fig1.add_shape(
        go.layout.Shape(
            type="line",
            x0=median,
            x1=median,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(
                color="red",
                width=2,
                dash="dash"
            )
        )
    )
    fig1.add_annotation(
        x=median,
        y=max_+100,
        text=str(median),
        name="Median",
        showarrow=False,
        xshift=15,
        font=dict(
            color="red"
        ))

    fig1.add_trace(
        go.Scatter(
            x=[None],  # Pas de données pour x et y
            y=[None],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name=f"Médiane"
        )
    )

    fig1.update_layout(
        title="Distribution des Notes Moyennes",
        xaxis_title="Note Moyenne",
        yaxis_title="Fréquence",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            # y=0.99,
            y=1.02,
            xanchor="left",
            # x=0.01
            x=0.01
        )
    )

    fig1.show()

    # Graphique du total de films par décennie
    total_films = df.groupby(
        "cuts", observed=True
    ).size().reset_index(name="total_films")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=total_films["cuts"],
        y=total_films["total_films"],
        showlegend=False,
        marker=dict(
            color='royalblue',
            line=dict(color='black', width=1))
    ))
    median = total_films["total_films"].median()
    fig2.add_shape(
        go.layout.Shape(
            type="line",
            x0=0,
            x1=1,
            y0=median,
            y1=median,
            xref="paper",
            line=dict(
                color="red",
                width=2,
                dash="dash"
            )
        )
    )
    fig2.add_annotation(
        x=-0.99,
        y=median,
        text=str(median),
        showarrow=False,
        yshift=10,
        # xshift=-10,
        font=dict(
            color="red"
        ))

    fig2.add_trace(
        go.Scatter(
            x=[None],  # Pas de données pour x et y
            y=[None],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name=f"Médiane",
        )
    )

    fig2.update_layout(
        title="Total des Films par Décénnie",
        xaxis_title="Année",
        yaxis_title="Quantité de Film Produit",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            # y=0.99,
            y=1.02,
            xanchor="left",
            # x=0.01
            x=0.01
        )
    )
    fig2.show()

    # Graphique du total de votes par décennie
    rating_votes = df.groupby(
        "cuts",
        observed=True)["rating_votes"].mean().reset_index(
            name="votes"
        )
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=rating_votes["cuts"],
        y=rating_votes["votes"],
        showlegend=False,
        marker=dict(
            color='royalblue',
            line=dict(color='black', width=1)
        )
    ))

    quantiles = rating_votes["votes"].quantile(
        [0.25, 0.5, 0.75]).values
    colors = [
        ("#065535", "1"),
        ("#ff0000", "2"),
        ("#b37400", "3")
    ]
    for q, color in zip(quantiles, colors):
        fig3.add_shape(
            go.layout.Shape(
                type="line",
                x0=0,
                x1=1,
                y0=q,
                y1=q,
                xref="paper",
                line=dict(
                    color=color[0],
                    width=2,
                    dash="dash"
                )
            )
        )
        fig3.add_annotation(
        x=-0.99,
        y=q,
        text=str(round(q)),
        showarrow=False,
        yshift=10,
        font=dict(
            color=color[0]
        ))

        fig3.add_trace(
            go.Scatter(
                x=[None],  # Pas de données pour x et y
                y=[None],
                mode="lines",
                line=dict(color=color[0], width=2, dash="dash"),
                name=f"Quantile {color[1]}",
            )
        )

    fig3.update_layout(
        title="Total des Votes par Décénnie",
        xaxis_title="Année",
        yaxis_title="Quantité de Votes",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01
        )
    )

    # fig3.update_layout(
    #     title="Total des Votes par Décénnie",
    #     xaxis_title="Année",
    #     yaxis_title="Quantité de Votes"
    # )
    fig3.show()
