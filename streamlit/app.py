import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Supprime les boutons full screen des images de l'app.
hide_img_fs = '''
                        <style>
                        button[title="View fullscreen"]{
                            visibility: hidden;
                        }
                        </style>
                    '''
st.markdown(hide_img_fs, unsafe_allow_html=True)

# Importation des dataframes nécessaires.
machine_learning = "datasets/machine_learning_final.parquet"
site_web = "datasets/site_web.parquet"

df_machine_learning = pd.read_parquet(machine_learning)
df_site_web = pd.read_parquet(site_web)

# Création de la liste des films pour la sélection.
movies = df_site_web["titre_str"]

def combine(r):
    return (
        r["keywords"]
        + " "
        + r["actors"]
        + " "
        + r["director"]
        +" "
        +r["titre_genres"]
)


df_machine_learning["one_for_all"] = df_machine_learning.apply(combine, axis=1)

def get_info(
        df: pd.DataFrame,
        info_type: str
    ):
    """
    Récupère les infos demandées sur le film selectionné.

    Paramètres :
    selected_movie : pd.DataFrame : DataFrame dans lequel rechercher l'info.
    info_type : str : Type d'info demandé.
    """
    info = df[info_type].iloc[0]
    return info

def idx_titre(
        df: pd.DataFrame,
        idx: int
):
    return df[df.index == idx]["titre_str"].values[0]

def knn_algo(selectvalue
):
    index = df_machine_learning[df_machine_learning["titre_str"] == selectvalue].index[0]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df_machine_learning["one_for_all"])
    knn_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    ).fit(count_matrix)
    dist, indices = knn_model.kneighbors(
        count_matrix[index], n_neighbors = 6
    )
    result = []
    for idx, dis in zip(indices.flatten()[1:], dist.flatten()[1:]):
        recommandations = idx_titre(df_machine_learning, idx)
        result.append(recommandations)
    return result


# Début de la page.
st.header(
    "DigitalDreamers Recommandation System",
    anchor = False
)
default_message = "Entrez ou sélectionnez le nom d'un film..."
# Barre de sélection de films.
selectvalue = st.selectbox(
    label = "Choisissez un film ⤵️",
    options = [default_message] + list(sorted(movies)),
    placeholder = default_message,
)

if selectvalue != default_message:
    # st.experimental_set_query_params(movie=selectvalue)
    # Bouton de recommandation de films similaires.
    recommendations_button = st.button(
        "💡 Recommandations 💡"
        )
    selected_movie = df_site_web[df_site_web["titre_str"] == selectvalue]
    # st.link_button("bande annonce", url = "http://localhost:8501/#bande-annonce")
    if recommendations_button:
        col1, col2, col3, col4, col5 = st.columns(5)
        movies_recommandations = knn_algo(selectvalue)
        print(movies_recommandations)
        col = (
            (col1, movies_recommandations[0]),
            (col2, movies_recommandations[1]),
            (col3, movies_recommandations[2]),
            (col4, movies_recommandations[3]),
            (col5, movies_recommandations[4])
        )
        for i in col:
            movie = df_machine_learning[df_machine_learning["titre_str"] == col[1]]
            colonne = col[0]
            image_link = get_info(movie, "image")
            colonne.image(image_link)

    else:
        # Affichage des infos du film sélectionné.
        col1, col2 = st.columns([1, 1])
        image_link = get_info(selected_movie, "image")
        col1.image(image_link, width = 325, use_column_width = "always")
        with col2:
            date = get_info(selected_movie, "date")
            titre = get_info(selected_movie, "titre_str")
            # Titre + Date de sortie du film sélectionné.
            st.header(
                f"{titre} - ({date})",
                anchor = False,
                divider = True
            )
            director_name = get_info(selected_movie, "director")
            actors_list = get_info(selected_movie, "actors")
            genre_list = get_info(selected_movie, "titre_genres")
            overview = get_info(selected_movie, "overview")
            # Affichage des genres du film.
            st.caption(
                f"<p style='font-size: 16px;'>{genre_list}</p>",
                unsafe_allow_html=True
            )
            # Affichage du réalisateur du film.
            st.subheader(
                f"**Réalisateur :**",
                anchor = False,
                divider = True
            )
            st.markdown(
                f"{director_name}",
                unsafe_allow_html=True)
            # Affichage des acteurs principaux du film.
            st.subheader(
                f"**Acteurs :**",
                anchor = False,
                divider = True
            )
            st.markdown(f"{actors_list}")
        # Affichage du résumé du film.
        st.subheader(
                f"**Synopsis :**",
                anchor = False,
                divider = True
            )
        st.markdown(f"{overview}")
        # Affichage de la bande annonce du film.
        st.subheader(
                f"**Bande Annonce :**",
                # anchor = False,
                divider = True
            )
        video_link = get_info(selected_movie, "youtube")
        st.video(video_link)