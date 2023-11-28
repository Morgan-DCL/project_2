import pandas as pd
import numpy as np
import requests

import aiohttp
import asyncio

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from st_click_detector import click_detector

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
        "date", 'popularity', 'rating_avg', 'rating_vote'
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
    # director: bool = False
):
    index = 1
    peo = api_list[nb]
    width = 125 #125
    height = 180 # 180
    actor_actress = 'Acteur' if peo["gender"] == 2 else 'Actrice'
    # import pprint
    # pprint.pprint(peo)
    # <a href="{peo['biography']}" target="_self">{peo['biography']}<id="{api_list[nb]}">

    # <p style="margin: 0;">{'Réalisateur' if nb < 1 else actor_actress}</p>
    content = f"""
        <div style="text-align: center;">
            <a href="full_bio" target="./full_bio"><id="{api_list[nb]}">
                <img width="{str(width)}px" height="{str(height)}px" src="{peo['image']}"
                    style="object-fit: cover; border-radius: 5%; margin-bottom: 15px;">
            </a>

            <p style="margin: 0;">{actor_actress}</p>
            <p style="margin: 0;"><strong>{peo['name']}</strong></p>
        </div>
    """
    unique_key = f"click_detector_{np.random.random()}"
    return index, click_detector(content, key=unique_key)

# @st.cache_data
def afficher_details_film(df: pd.DataFrame):
    # from PIL import Image
    # import base64
    # from io import BytesIO

    # buffered = BytesIO()
    # image = Image.open('fm6KqXpk3M2HVveHwCrBSSBaO0V.jpg')
    # image.save(buffered, format="JPEG")
    # img_str = base64.b64encode(buffered.getvalue()).decode()

    infos = {
        "date": get_info(df, "date"),
        "image": get_info(df, "image"),
        "titre_str": get_info(df, "titre_str"),
        "titre_genres": get_info(df, "titre_genres"),
        "rating_avg": get_info(df, "rating_avg"),
        "rating_vote": get_info(df, "rating_vote"),
        "popularity": get_info(df, "popularity"),
        "runtime": get_info(df, "runtime"),
        "synopsis": get_info(df, "overview"),
        "tagline": get_info(df, "tagline"),
    }
    film_str : str = infos["titre_str"]
    name_film = film_str[:-7] if film_str.endswith(")") else film_str

    # background_style = f"""
    #     background-image: url('data:image/jpeg;base64,{img_str}');
    #     background-size: cover;
    #     background-repeat: no-repeat;
    #     background-attachment: fixed;
    #     position: relative;
    #     z-index: 1;
    # """

    # overlay_style = """
    #     position: absolute;
    #     top: 0;
    #     left: 0;
    #     right: 0;
    #     bottom: 0;
    #     background-color: rgba(0, 0, 0, 0.5); /* Overlay foncé avec une opacité */
    #     z-index: 2;
    # """

    # st.markdown(f'<div style="{background_style}">', unsafe_allow_html=True)
    # st.markdown(f'<div style="{overlay_style}"></div>', unsafe_allow_html=True)

    col1, col2, cols3 = st.columns([1, 2, 1])

    with col1:
        st.image(infos["image"], use_column_width=True)

    runtime = infos["runtime"]
    actors_list = [a for a in get_actors_dict(df).values()]
    director_list = [d for d in get_directors_dict(df).values()]
    director = asyncio.run(fetch_persons_bio(director_list, True))
    actors = asyncio.run(fetch_persons_bio(actors_list))

    with col2:
        cap, cap1 = st.columns(2)
        with cap:
            st.header(
            f"{name_film} - ({infos['date']})", anchor=False, divider=True)

            st.caption(
                f"<p style='font-size: 16px;'>{infos['titre_genres']}  •  {f'{runtime // 60}h {runtime % 60}m'}</p>",
                unsafe_allow_html=True
            )
            color_rating = "#198c19" if infos["rating_avg"] >= 7 else "#F4D03F" if infos["rating_avg"] >= 5 else "#E74C3C"
            color_vote = "#967259"
            color_popularity = "#967259"
            txt_color = "#F2F2F2"

            gap = 0.1

            elements_html = f"""
                <div style="display: flex; flex-direction: column; align-items: center; gap: {gap}px;">
                    <p>Notes</p>
                    <div style="background-color: {color_rating}; border-radius: 50%; width: 60px; height: 60px;">
                        <h2 style="text-align: center; color: {txt_color}; font-size: 22px;">{round(infos["rating_avg"], 2)}</h2>
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; align-items: center; gap: {gap}px;">
                    <p>Votes</p>
                    <div style="background-color: {color_vote}; border-radius: 50%; width: 60px; height: 60px;">
                        <h2 style="text-align: center; color: {txt_color}; font-size: 22px;">{round(infos["rating_vote"])}</h2>
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; align-items: center; gap: {gap}px;">
                    <p>Popularité</p>
                    <div style="background-color: {color_popularity}; border-radius: 50%; width: 60px; height: 60px;">
                        <h2 style="text-align: center; color: {txt_color}; font-size: 22px;">{round(infos["popularity"])}</h2>
                    </div>
                </div>
            """
            st.markdown(f"<div style='display: flex; justify-content: start; gap: 20px;'>{elements_html}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            texte_fondu = f'<span style="color: #555;">*{infos["tagline"]}*</span>'
            st.write(texte_fondu, unsafe_allow_html=True)

        width = 125
        height = 180

        with cap1:
            st.header(
            f"Réalisateur", anchor=False, divider=True)

                    # <p style="margin: 0;">Réalisateur</p>

            content = f"""
                <div style="text-align: center;">
                        <img width="{str(width)}px" height="{str(height)}px" src="{director[0]['image']}"
                            style="object-fit: cover; border-radius: 5%; margin-bottom: 15px;">
                    <p style="margin: 0;"><strong>{director[0]['name']}</strong></p>
                </div>
            """
            click_detector(content, key=np.random.random())


        st.subheader("**Distribution :**",anchor=False, divider=True)
        # one_for_all = director + actors
        # one_for_all = director + actors
        cols = st.columns(len(actors))
        for i, col in enumerate(cols):
            with col:
                index, clicked = get_clicked_act_dirct(actors, i)

    with cols3:
        st.header("**Bande Annonce :** ", anchor=False, divider=True)
        st.video(get_info(df, "youtube"))

        st.subheader("**Synopsis :**", anchor=False, divider=True)
        st.markdown(infos["synopsis"])


    st.markdown('</div>', unsafe_allow_html=True)

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
