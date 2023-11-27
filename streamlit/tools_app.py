import pandas as pd
import numpy as np
import requests

import aiohttp
import asyncio

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from st_click_detector import click_detector

# from tools import import_config, logging
import streamlit as st
import streamlit.components.v1 as components

async def fetch_infos(
    ss: object,
    TMdb_id: int,
):
    params = {
        "api_key": "fe4a6f12753fa6c12b0fc0253b5e667f",
        "include_adult": "False",
        "language": "fr-FR",
        "append_to_response": "combined_credits",
    }
    base_url = "https://api.themoviedb.org/3/person/"
    url = f"{base_url}{TMdb_id}"
    async with ss.get(url, params=params) as rsp:
        return await rsp.json()

async def fetch_persons_bio(
    people_list: list, director: bool = False
) -> list:
    url_image = "https://image.tmdb.org/t/p/w300_and_h450_bestv2"
    async with aiohttp.ClientSession() as ss:
        taches = []
        for id in people_list:
            tache = asyncio.create_task(
                fetch_infos(ss, id)
            )
            taches.append(tache)
        datas = await asyncio.gather(*taches)
        full = []
        for data in datas:
            data["image"] = f"{url_image}{data['profile_path']}"
            # 99: Documentaire, 16: Animation, 10402: Musique
            exclude = [99, 10402] if director else [99, 16, 10402]
            if director:
                top_credits = sorted(
                    (
                        n for n in data["combined_credits"]["crew"]
                        if n["media_type"] == "movie" and n["job"] =="Director"
                        and all(genre not in n["genre_ids"] for genre in exclude)
                    ),
                    key=lambda x: (-x['popularity'], -x['vote_average'], -x["vote_count"])
                )[:8]
            else:
                top_credits = sorted(
                    (
                        n for n in data["combined_credits"]["cast"]
                        if n["media_type"] == "movie" and n["order"] <= 3
                        and all(genre not in n["genre_ids"] for genre in exclude)
                    ),
                    key=lambda x: (-x['popularity'], -x['vote_average'], -x["vote_count"])
                )[:8]
            data["top_5"] = [n["title"] for n in top_credits]
            data["top_5_images"] = [f"{url_image}{n['poster_path']}" for n in top_credits]
            data["top_5_movies_ids"] = [n['id'] for n in top_credits]
            to_pop = (
                "adult",
                "also_known_as",
                # "gender",
                "homepage",
                "profile_path",
                "combined_credits",
                "known_for_department",
            )
            for tp in to_pop:
                data.pop(tp)
            full.append(data)
    return full


def clean_dup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les doublons dans une colonne spécifique d'un DataFrame en ajoutant
    la date entre parenthèses.

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame à nettoyer.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les doublons nettoyés.
    """
    condi = df["titre_str"].duplicated(keep=False)
    df.loc[condi, "titre_str"] = (
        df.loc[
            condi, "titre_str"
        ] + " " + "(" + df.loc[condi, "date"].astype(str) + ")"
    )
    return df

def auto_scroll():
    """
    Déclenche un défilement automatique de la fenêtre dans un contexte Streamlit.

    Cette fonction ne prend aucun paramètre et ne retourne rien.
    Elle utilise un script HTML pour réinitialiser le défilement de la fenêtre.
    """
    components.html(
        f"""
            <p>{st.session_state["counter"]}</p>
            <script>
                window.parent.document.querySelector('section.main').scrollTo(0, 0);
            </script>
        """,
        height=0
    )

def get_info(
        df: pd.DataFrame,
        info_type: str
    ):
    """
    Extrait une information spécifique du premier élément d'une colonne d'un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    info_type : str
        Le nom de la colonne dont on extrait l'information.

    Returns
    -------
    Any
        Information extraite de la première ligne de la colonne spécifiée.
    """
    info = df[info_type].iloc[0]
    return info


def get_titre_from_index(
        df: pd.DataFrame,
        idx: int
    ) -> str:
    """
    Récupère le titre correspondant à un index donné dans un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les titres.
    idx : int
        Index du titre à récupérer.

    Returns
    -------
    str
        Titre correspondant à l'index fourni.
    """
    return df[df.index == idx]["titre_str"].values[0]

def get_index_from_titre(
        df: pd.DataFrame,
        titre: str
    ) -> int:
    """
    Trouve l'index correspondant à un titre donné dans un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les titres.
    titre : str
        Le titre dont on cherche l'index.

    Returns
    -------
    int
        Index du titre dans le DataFrame.
    """
    return df[df.titre_str == titre].index[0]

def knn_algo(df: pd.DataFrame, titre: str, top: int = 5) -> list:
    """
    Implémente l'algorithme KNN pour recommander des titres similaires.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données pour le modèle KNN.
    titre : str
        Titre à partir duquel les recommandations sont faites.

    Returns
    -------
    List[str]
        Liste de titres recommandés.
    """
    index = df[
        df["titre_str"] == titre
    ].index[0]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(
        df["one_for_all"]
    )
    knn_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    ).fit(count_matrix)
    dist, indices = knn_model.kneighbors(
        count_matrix[index], n_neighbors = top+1
    )
    result = []
    for idx, dis in zip(indices.flatten()[1:], dist.flatten()[1:]):
        recommandations = get_titre_from_index(df, idx)
        result.append(recommandations)
    return result

def infos_button(df: pd.DataFrame, movie_list: list, idx: int):
    """
    Met à jour une variable de session Streamlit en fonction de l'index du film sélectionné.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les informations des films.
    movie_list : list
        Liste des titres de films.
    idx : int
        Index du film sélectionné.

    Cette fonction ne retourne rien mais met à jour la variable de session "index_movie_selected".
    """
    titre = get_titre_from_index(df, idx)
    st.session_state["index_movie_selected"] = movie_list.index(titre)

def get_clicked(
    df: pd.DataFrame,
    titres_list: list,
    nb: int,
    genre: str = "Drame",
    key_: bool = False
):
    """
    Génère un élément cliquable pour un film et renvoie son index et un détecteur de clic.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les films.
    titres_list : list
        Liste des titres de films.
    nb : int
        Numéro du film dans la liste.
    genre : str, optional
        Genre du film, par défaut à "Drame".
    key_ : bool, optional
        Si vrai, génère une clé unique pour le détecteur de clic, par défaut à False.

    Returns
    -------
    Tuple[int, Any]
        Index du film et l'objet du détecteur de clic.
    """
    index = int(get_index_from_titre(df, titres_list[nb]))
    movie = df[df["titre_str"] == titres_list[nb]]
    image_link = get_info(movie, "image")
    content = f"""<a href="#" id="{titres_list[nb]}"><img width="125px" heigth="180px" src="{image_link}" style="border-radius: 5%"></a>"""
    if key_:
        unique_key = f"click_detector_{genre}_{index}"
        return index, click_detector(content, key=unique_key)
    else:
        return index, click_detector(content)



@st.cache_data
def afficher_top_genres(df: pd.DataFrame, genres: str) -> pd.DataFrame:
    """
    Affiche les films les mieux classés d'un genre spécifique, excluant "Animation" sauf si spécifié.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les films.
    genres : str
        Genre de films à afficher.

    Returns
    -------
    pd.DataFrame
        DataFrame des films triés par popularité, note moyenne, et nombre de votes.
    """
    sort_by = [
        'popularity', 'rating_avg', 'rating_vote'
    ]
    ascending_ = [False for i in range(len(sort_by))]
    condi = (
        (
            df["titre_genres"].str.contains(genres) &
            ~df["titre_genres"].str.contains("Animation")
        )
        if genres != "Animation"
        else df["titre_genres"].str.contains(genres)
    )
    return df[condi].sort_values(by=sort_by, ascending=ascending_)

def get_clicked_act_dirct(
    api_list: list,
    nb: int,
):
    index = 1
    peo = api_list[nb]
    width = 125
    height = 180
    actor_actress = 'Acteur' if peo["gender"] == 2 else 'Actrice'
    import pprint
    pprint.pprint(peo)
    content = f"""
        <div style="text-align: center;">
            <a href="{peo['biography']}" target="_blank" id="{api_list[nb]}">
                <img width="{str(width)}px" height="{str(height)}px" src="{peo['image']}"
                    style="object-fit: cover; border-radius: 5%; margin-bottom: 15px;">
            </a>
            <p style="margin: 0;">{'Directeur' if nb < 1 else actor_actress}</p>
            <p style="margin: 0;"><strong>{peo['name']}</strong></p>
        </div>
    """
    unique_key = f"click_detector_{np.random.random()}"
    return index, click_detector(content, key=unique_key)



def afficher_details_film(config: dict, df: pd.DataFrame):
    """
    Affiche les détails d'un film dans une interface Streamlit, incluant l'image et des informations clés.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les informations du film.

    Cette fonction ne retourne rien mais utilise Streamlit pour afficher des détails tels que
    le titre, le genre, le réalisateur, et les acteurs du film.
    """
    col1, col2 = st.columns([1, 3])
    col1.image(get_info(df, "image"), use_column_width = True)
    columns = [
        "titre_str",
        "titre_genres",
        "overview",
        "actors",
    ]
    runtime = get_info(df, "runtime")
    film_str = get_info(df, "titre_str")
    name_film = film_str if not film_str.__contains__("(") else film_str[:-7]
    with col2:
        for detail in columns:
            if detail == "titre_str":
                st.header(
                    f"{name_film} - ({get_info(df, 'date')})", anchor=False, divider=True)
            if detail == "titre_genres":
                st.caption(
                    f"<p style='font-size: 16px;'>{get_info(df, detail)}  •  {f'{runtime // 60}h {runtime % 60}m'}</p>", unsafe_allow_html=True)
                print(df.columns)
                rating_avg = get_info(df, 'rating_avg')
                # "#58D68D"
                color = "#198c19" if rating_avg >= 7 else "#F4D03F" if rating_avg >= 5 else "#E74C3C"
                circle_html = f"""
                <div style="display: inline-flex; align-items: center; justify-content: center; background-color: {color};
                            border-radius: 50%; width: 60px; height: 60px;">
                <h2 style="text-align: center; color: #F2F2F2; font-size: 25px;">{round(rating_avg, 2)}</h2>
                </div>
                """
                st.markdown(circle_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                texte_italique = f"*{get_info(df, 'tagline')}*"
                texte_fondu = f'<span style="color: #555;">{texte_italique}</span>'
                st.write(texte_fondu, unsafe_allow_html=True)

            if detail == "overview":
                st.subheader("**Synopsis :**", anchor=False, divider=True)
                st.markdown(get_info(df, detail))
            # if detail == "actors":
            #     st.subheader("",anchor=False, divider=True)
            #     actors_list = [a for a in get_actors_dict(df).values()]
            #     director_list = [d for d in get_directors_dict(df).values()]
            #     director = asyncio.run(fetch_persons_bio(config, director_list, True))
            #     actors = asyncio.run(fetch_persons_bio(config, actors_list))
            #     one_for_all = director + actors
            #     cols = st.columns(len(one_for_all))
            #     for i, col in enumerate(cols):
            #         with col:
            #             print(i)
            #             index, clicked = get_clicked_act_dirct(one_for_all, i)
                # Une fois cliqué créér une page uniquement de biographie avec leurs films etc ....
                # Quand on clique sur un de leurs films, go back to case départ et a nouveau recommandation etc..

                # col1, col2 = st.columns([1, 3])
                # with col1:
                #     for n in actors:
                #         st.markdown(f"**{n['name']}**")
                # with col2:
                #         st.image([n["image"] for n in actors], width=75)
                        # st.image(actors_img, width=75)
            # else:
            #     st.subheader(f"**{detail.capitalize()} :**", anchor=False, divider=True)
            #     if detail == "director":
            #         director_list = [a for a in get_directors_dict(df).values()]
            #         director = asyncio.run(fetch_persons_bio(config, director_list, True))
            #         # col1, col2 = st.columns([3, 1])
            #         # with col1:
            #         #     for n in director:
            #         #         st.markdown(f"**{n['name']}**")
            #         # with col2:
            #         #     st.image([n["image"] for n in director], width=75)
            #         for n in director:
            #             cols = st.columns([1, 3])  # Crée une colonne plus large pour l'image
            #             with cols[0]:
            #                 st.image(n["image"], use_column_width=True)  # Centrer l'image
            #             with cols[1]:
            #                 st.write(f"**{n['name']}**")  # Le nom sous l'image
            #     if detail == "actors":
            #         actors_list = [a for a in get_actors_dict(df).values()]
            #         # actors = asyncio.run(fetch_persons_bio(config, actors_list))
            #         # col1, col2 = st.columns([3, 1])
            #         # with col1:
            #         #     for n in actors:
            #         #         st.markdown(f"**{n['name']}**")
            #         # with col2:
            #         #     st.image([n["image"] for n in actors], width=75)
            #     # Pour le réalisateur
            #     # director = asyncio.run(fetch_persons_bio(config, director_list, True))
            #         actors = asyncio.run(fetch_persons_bio(config, actors_list))
            #         for n in actors:
            #             cols = st.columns([1, 3])  # Crée une colonne plus large pour l'image
            #             with cols[0]:
            #                 st.image(n["image"], use_column_width=True)  # Centrer l'image
            #             with cols[1]:
            #                 st.write(f"**{n['name']}**")  # Le nom sous l'image
            #         # Pour les acteurs


def get_actors_dict(df: pd.DataFrame) -> dict:
    """
    Extrait un dictionnaire d'acteurs depuis un DataFrame.

    Cette fonction parcourt un DataFrame et construit un dictionnaire où les
    clés sont les noms des acteurs et les valeurs sont leurs identifiants.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant deux colonnes : 'actors' et 'actors_ids'.
        'actors' est une chaîne de caractères avec des noms d'acteurs séparés
        par des virgules, et 'actors_ids' sont les identifiants correspondants.

    Returns
    -------
    dict
        Dictionnaire où les clés sont les noms des acteurs et les valeurs
        sont les identifiants correspondants.
    """
    actors_dict = {}
    for actors, ids in zip(df.actors, df.actors_ids):
        actors_list = actors.split(", ")
        actor_id_pairs = zip(actors_list, ids)
        actors_dict.update(actor_id_pairs)
    return actors_dict


def get_directors_dict(df: pd.DataFrame) -> dict:
    """
    Extrait un dictionnaire de réalisateurs depuis un DataFrame.

    Cette fonction parcourt un DataFrame et construit un dictionnaire où les
    clés sont les noms des réalisateurs et les valeurs sont leurs identifiants.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant deux colonnes : 'director' et 'director_ids'.
        'director' est une chaîne de caractères avec des noms de réalisateurs
        séparés par des virgules, et 'director_ids' sont les identifiants
        correspondants.

    Returns
    -------
    dict
        Dictionnaire où les clés sont les noms des réalisateurs et les valeurs
        sont les identifiants correspondants.
    """
    directors_dict = {}
    for directors, ids in zip(df.director, df.director_ids):
        directors_list = directors.split(", ")
        directors_id_pairs = zip(directors_list, ids)
        directors_dict.update(directors_id_pairs)
    return directors_dict
