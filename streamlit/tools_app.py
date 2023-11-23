import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from st_click_detector import click_detector

import streamlit as st
import streamlit.components.v1 as components


def fetch_actors_bio(imdb_id: int):
    api_key = "fe4a6f12753fa6c12b0fc0253b5e667f"
    language = "fr-FR"
    params = {
        "api_key": api_key,
        "include_adult": "False",
        "language": language,
        "append_to_response": "combined_credits",
    }
    base_url = "https://api.themoviedb.org/3/person/"
    url_image = "https://image.tmdb.org/t/p/w300_and_h450_bestv2"
    url_youtube = "https://www.youtube.com/watch?v="
    url = f"{base_url}{imdb_id}"
    r = requests.get(url, params=params)
    data = r.json()

    data["image"] = f"{url_image}{data['profile_path']}"
    top_credits = sorted(
        (
            n for n in data["combined_credits"]["cast"]
            if n["media_type"] == "movie" and n["order"] <= 3
            and all(genre not in n["genre_ids"] for genre in [99, 16, 10402]) # 99: Documentaire, 16: Animation, 10402: Musique
        ),
        key=lambda x: (-x['popularity'], -x['vote_average'], -x["vote_count"])
    )[:8]
    data["top_5"] = [n["title"] for n in top_credits]
    data["top_5_images"] = [f"{url_image}{n['poster_path']}" for n in top_credits]
    data["top_5_TMDb_ids"] = [n['id'] for n in top_credits]

    to_pop = (
        "adult",
        "also_known_as",
        "gender",
        "homepage",
        "profile_path",
        "combined_credits",
        "known_for_department",
    )
    for tp in to_pop:
        data.pop(tp)
    return data



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

def knn_algo(df: pd.DataFrame, titre: str) -> list:
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
        count_matrix[index], n_neighbors = 6
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

def get_clicked_act_dirct(
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

def afficher_details_film(df: pd.DataFrame):
    """
    Affiche les détails d'un film dans une interface Streamlit, incluant l'image et des informations clés.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les informations du film.

    Cette fonction ne retourne rien mais utilise Streamlit pour afficher des détails tels que
    le titre, le genre, le réalisateur, et les acteurs du film.
    """
    col1, col2 = st.columns([1, 1])
    col1.image(get_info(df, "image"), width=425, use_column_width="always")
    columns = [
        "titre_str",
        "titre_genres",
        "director",
        "actors"
    ]
    with col2:
        for detail in columns:
            if detail == "titre_str":
                st.header(
                    f"{get_info(df, detail)} - ({get_info(df, 'date')})", anchor=False, divider=True)
            elif detail == "titre_genres":
                st.caption(
                    f"<p style='font-size: 16px;'>{get_info(df, detail)}</p>", unsafe_allow_html=True)
            else:
                st.subheader(f"**{detail.capitalize()} :**", anchor=False, divider=True)
                if detail == "director":
                    director_dict = get_directors_dict(df)
                    # for director_name, directors_id in director_dict.items():
                    #     col1, col2 = st.columns([1, 1])
                    #     with col1:
                    #         st.markdown(f"**{director_name}**")
                    #     with col2:
                    #         st.image(fetch_actors_bio(directors_id)["image"], width=50)
                    for n in list(get_directors_dict(df)):
                        st.markdown(n)
                    directors = get_directors_dict(df).keys()
                    directors_id = get_directors_dict(df).values()
                if detail == "actors":
                    actors_dict = get_actors_dict(df)
                    for actor_name, actor_id in actors_dict.items():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{actor_name}**")
                        with col2:
                            st.image(fetch_actors_bio(actor_id)["image"], width=50)


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
