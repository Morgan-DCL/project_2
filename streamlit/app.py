import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from st_click_detector import click_detector

import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="DigitalDreamers Recommandation System",
    page_icon="📽️",
    initial_sidebar_state="collapsed",
)
# Supprime les boutons fullscreen des images de l'app.
hide_img_fs = """
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    </style>
"""
st.markdown(hide_img_fs, unsafe_allow_html=True)
# Arrondi les coins des images.
round_corners = """
    <style>
        .st-emotion-cache-1v0mbdj > img{
            border-radius:2%;
        }
    </style>
"""
st.markdown(round_corners, unsafe_allow_html=True)

st.markdown(
    """
    <script>
        function streamlit_on_click(index) {
            const buttonCallbackManager = Streamlit.ButtonCallbackManager.getManager();
            buttonCallbackManager.setCallback("infos_button", index => {
                Streamlit.setComponentValue(index);
            });
            buttonCallbackManager.triggerCallback("infos_button", index);
        }
    </script>
    """,
    unsafe_allow_html=True,
)

# Importation des dataframes nécessaires.
machine_learning = "datasets/machine_learning_final.parquet"
site_web = "datasets/site_web.parquet"
df_machine_learning = pd.read_parquet(machine_learning)
df_site_web = pd.read_parquet(site_web)

# Création de la liste des films pour la sélection.
default_message = "Entrez ou sélectionnez le nom d'un film..."
movies = df_site_web["titre_str"]
movies_list = [default_message] + list(sorted(movies))
selectvalue = default_message


# Création de la colonne "one_for_all" (TEMPORAIRE)
def combine(r):
    return (
        r["keywords"]
        + " "
        + r["actors"]
        + " "
        + r["director"]
        + " "
        + r["titre_genres"]
    )


# Ajout de la colonne sur le df_machine_learning
df_machine_learning["one_for_all"] = df_machine_learning.apply(
    combine, axis=1
)


# Fonctions utilisées par l'app.
def get_info(df: pd.DataFrame, info_type: str):
    """
    Récupère les infos demandées sur le film selectionné dans un dataframe
    déjà filtré.
    ---
    Paramètres :
    selected_movie : pd.DataFrame : DataFrame contenant un seul film.
    info_type : str : Type d'info demandé.
    ---
    Retourne :
    La valeur de l'info demandée.
    ---
    Exemple :
    Lien.jpg,
    'titre' en str,
    lien de vidéo youtube, ...
    """
    info = df[info_type].iloc[0]
    return info


def get_titre_from_index(df: pd.DataFrame, idx: int):
    """
    Récupère le 'titre_str' à partir de l'index d'un film.
    ---
    Paramètres :
    df : pd.DataFrame : DataFrame dans lequel rechercher l'info.
    idx : int : Index du film recherché.
    ---
    Retourne :
    'titre_str' (str)
    """
    return df[df.index == idx]["titre_str"].values[0]


def get_index_from_titre(df: pd.DataFrame, titre: str):
    """
    Récupère l'index à partir du 'titre_str' d'un film.
    ---
    Paramètres :
    df : pd.DataFrame : DataFrame dans lequel rechercher l'info.
    titre : str : Titre du film recherché.
    ---
    Retourne :
    Index du film (int)
    """
    return df[df.titre_str == titre].index[0]


def knn_algo(selectvalue):
    """
    Algorithme récupérant une liste contenant le 'titre_str'
    des 5 films recommandés à partir du 'titre_str' d'un film
    sélectionné.
    ---
    Retourne :
    Titre des 5 films recommandés (list).
    """
    index = df_machine_learning[
        df_machine_learning["titre_str"] == selectvalue
    ].index[0]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df_machine_learning["one_for_all"])
    knn_model = NearestNeighbors(metric="cosine", algorithm="brute").fit(
        count_matrix
    )
    dist, indices = knn_model.kneighbors(
        count_matrix[index], n_neighbors=6
    )
    result = []
    for idx, dis in zip(indices.flatten()[1:], dist.flatten()[1:]):
        recommandations = get_titre_from_index(df_machine_learning, idx)
        result.append(recommandations)
    return result


# Bouton "Plus d'infos..." lors de la recommandation.


def infos_button(index):
    """
    Récupère l'index d'un film et change le film sélectionné sur
    la page par le titre de celui-ci
    ---
    Paramètres :
    index : index du film recherché.
    ---
    Retourne :
    Change l'index du film sélectionné dans la session_state : 'index_movie_selected'.
    """
    titre = get_titre_from_index(df_site_web, index)
    st.session_state["index_movie_selected"] = movies_list.index(titre)


def get_clicked(
    df,
    titres_list,
    nb,
):
    index = int(get_index_from_titre(df, titres_list[nb]))
    movie = df[df["titre_str"] == titres_list[nb]]
    image_link = get_info(movie, "image")
    content = f"""<a href='#' id='{titres_list[nb]}'><img width='135px' heigth='180px' src="{image_link}"></a>"""
    return (index, click_detector(content))


header_anchor = "top"
# Début de la page.
st.header("DigitalDreamers Recommandation System", anchor=header_anchor)

# Instanciation des session_state.
if "index_movie_selected" not in st.session_state:
    st.session_state["index_movie_selected"] = movies_list.index(
        selectvalue
    )

# Barre de sélection de films.
selectvalue = st.selectbox(
    label="Choisissez un film ⤵️",
    options=movies_list,
    placeholder=default_message,
    index=st.session_state["index_movie_selected"],
)

if selectvalue != default_message:
    # Bouton de recommandation de films similaires.
    recommendations_button = st.button("Films similaires 💡")
    selected_movie = df_site_web[df_site_web["titre_str"] == selectvalue]
    # Quand le bouton recommandation est appuyé.
    if recommendations_button:
        # Affichage des images pour les 5 films recommandés.
        col1, col2, col3, col4, col5 = st.columns(5)
        recommended = knn_algo(selectvalue)
        image_cols = (
            (col1, recommended[0]),
            (col2, recommended[1]),
            (col3, recommended[2]),
            (col4, recommended[3]),
            (col5, recommended[4]),
        )
        for col in image_cols:
            movie = df_machine_learning[
                df_machine_learning["titre_str"] == col[1]
            ]
            colonne = col[0]
            image_link = get_info(movie, "image")
            colonne.image(image_link, width=135)
        # Affichage du bouton "Plus d'infos..." pour chaque films recommandés.
        col6, col7, col8, col9, col10 = st.columns(5)
        button_cols = (
            (col6, int(get_index_from_titre(df_site_web, recommended[0]))),
            (col7, int(get_index_from_titre(df_site_web, recommended[1]))),
            (col8, int(get_index_from_titre(df_site_web, recommended[2]))),
            (col9, int(get_index_from_titre(df_site_web, recommended[3]))),
            (
                col10,
                int(get_index_from_titre(df_site_web, recommended[4])),
            ),
        )
        for col in button_cols:
            index = col[1]
            col[0].button(
                "Plus d'infos...",
                on_click=infos_button,
                args=(col[1],),
                key=index,
            )
        st.button("🔼 Cacher")
    # Affichage des infos du film sélectionné.
    col1, col2 = st.columns([1, 1])
    image_link = get_info(selected_movie, "image")
    col1.image(image_link, width=325, use_column_width="always")
    with col2:
        date = get_info(selected_movie, "date")
        titre = get_info(selected_movie, "titre_str")
        # Titre + Date de sortie du film sélectionné.
        st.header(f"{titre} - ({date})", anchor=False, divider=True)
        director_name = get_info(selected_movie, "director")
        actors_list = get_info(selected_movie, "actors")
        genre_list = get_info(selected_movie, "titre_genres")
        overview = get_info(selected_movie, "overview")
        # Affichage des genres du film.
        st.caption(
            f"<p style='font-size: 16px;'>{genre_list}</p>",
            unsafe_allow_html=True,
        )
        # Affichage du réalisateur du film.
        st.subheader(f"**Réalisateur :**", anchor=False, divider=True)
        st.markdown(f"{director_name}", unsafe_allow_html=True)
        # Affichage des acteurs principaux du film.
        st.subheader(f"**Acteurs :**", anchor=False, divider=True)
        st.markdown(f"{actors_list}")
    # Affichage du résumé du film.
    st.subheader(f"**Synopsis :**", anchor=False, divider=True)
    st.markdown(f"{overview}")
    # Affichage de la bande annonce du film.
    st.subheader(f"**Bande Annonce :**", anchor=False, divider=True)
    video_link = get_info(selected_movie, "youtube")
    st.video(video_link)

else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Top 5 Films du moment :", anchor=False)
    col1, col2, col3, col4, col5 = st.columns(5)
    popularity = df_site_web.sort_values("popularity", ascending=False)
    titres = []
    for film in popularity["titre_str"].head():
        titres.append(film)
    with col1:
        index1, clicked1 = get_clicked(popularity, titres, 0)
    with col2:
        index2, clicked2 = get_clicked(popularity, titres, 1)
    with col3:
        index3, clicked3 = get_clicked(popularity, titres, 2)
    with col4:
        index4, clicked4 = get_clicked(popularity, titres, 3)
    with col5:
        index5, clicked5 = get_clicked(popularity, titres, 4)
    if clicked1:
        infos_button(index1)
        st.rerun()
    elif clicked2:
        infos_button(index2)
        st.rerun()
    elif clicked3:
        infos_button(index3)
        st.rerun()
    elif clicked4:
        infos_button(index4)
        st.rerun()
    elif clicked5:
        infos_button(index5)
        st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    # st.markdown("<br><br>", unsafe_allow_html=True)
    st.write("Comment utiliser l'application de recommandations :")
    st.write("1. Choisissez ou entrer le nom d'un film.")
    st.write(
        "2. Cliquez sur le bouton en haut de l'écran pour voir les films similaires."
    )
    st.write(
        "3. Cliquez sur une des recommandations pour avoir plus d'infos."
    )

st.write(
    "App développée par [Morgan](https://github.com/Morgan-DCL) et [Teddy](https://github.com/dsteddy)"
)
