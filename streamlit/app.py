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
default_message = "Entrez ou sélectionnez le nom d'un film..."

movies = df_site_web["titre_str"]
movies_list = [default_message] + list(sorted(movies))

selectvalue = default_message

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


df_machine_learning["one_for_all"] = df_machine_learning.apply(
    combine,
    axis=1
)

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

def get_titre_from_index(
        df: pd.DataFrame,
        idx: int
):
    return df[df.index == idx]["titre_str"].values[0]

def get_index_from_titre(
        df: pd.DataFrame,
        titre: str):
    return df[df.titre_str == titre].index[0]

def knn_algo(selectvalue):
    index = df_machine_learning[
        df_machine_learning["titre_str"] == selectvalue
    ].index[0]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(
        df_machine_learning["one_for_all"]
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
        recommandations = get_titre_from_index(df_machine_learning, idx)
        result.append(recommandations)
    return result

def handle_click(movie_title):
    titre = get_titre_from_index(df_site_web, movie_title)
    st.session_state["index_movie_selected"] = movies_list.index(titre)

# Début de la page.
st.header(
    "DigitalDreamers Recommandation System",
    anchor = False
)

if "index_movie_selected" not in st.session_state:
    st.session_state["index_movie_selected"] = movies_list.index(selectvalue)

# Barre de sélection de films.
selectvalue = st.selectbox(
    label = "Choisissez un film ⤵️",
    options = movies_list,
    placeholder = default_message,
    index = st.session_state["index_movie_selected"],
)

if selectvalue != default_message:
    # Bouton de recommandation de films similaires.
    recommendations_button = st.button(
        "💡 Recommandations 💡"
    )
    selected_movie = df_site_web[df_site_web["titre_str"] == selectvalue]
    if recommendations_button:
        col1, col2, col3, col4, col5 = st.columns(5)
        recommended = knn_algo(selectvalue)
        # col 1
        col1_movie = df_machine_learning[df_machine_learning["titre_str"] == recommended[0]]
        col1_image_link = get_info(col1_movie, "image")
        col1.image(col1_image_link, width = 135)
        col1_index = int(get_index_from_titre(df_site_web, recommended[0]))
        col1_button = col1.button("Plus d'infos...",
                                on_click = handle_click,
                                args = (col1_index,),
                                key = col1_index
                        )
        # col 2
        col2_movie = df_machine_learning[df_machine_learning["titre_str"] == recommended[1]]
        col2_image_link = get_info(col2_movie, "image")
        col2.image(col2_image_link, width = 135)
        col2_index = int(get_index_from_titre(df_site_web, recommended[1]))
        col2_button = col2.button("Plus d'infos...",
                                on_click = handle_click,
                                args = (col2_index,),
                                key = col2_index
                        )
        # col 3
        col3_movie = df_machine_learning[df_machine_learning["titre_str"] == recommended[2]]
        col3_image_link = get_info(col3_movie, "image")
        col3.image(col3_image_link, width = 135)
        col3_index = int(get_index_from_titre(df_site_web, recommended[2]))
        col3_button = col3.button("Plus d'infos...",
                                on_click = handle_click,
                                args = (col3_index,),
                                key = col3_index
                        )
        # col 4
        col4_movie = df_machine_learning[df_machine_learning["titre_str"] == recommended[3]]
        col4_image_link = get_info(col4_movie, "image")
        col4.image(col4_image_link, width = 135)
        col4_index = int(get_index_from_titre(df_site_web, recommended[3]))
        col4_button = col4.button("Plus d'infos...",
                                on_click = handle_click,
                                args = (col4_index,),
                                key = col4_index
                        )
        # col 5
        col5_movie = df_machine_learning[df_machine_learning["titre_str"] == recommended[4]]
        col5_image_link = get_info(col5_movie, "image")
        col5.image(col5_image_link, width = 135)
        col5_index = int(get_index_from_titre(df_site_web, recommended[4]))
        col5_button = col5.button("Plus d'infos...",
                                on_click = handle_click,
                                args = (col5_index,),
                                key = col5_index
                        )


        # cols = (
        #     (col1, recommended[0]),
        #     (col2, recommended[1]),
        #     (col3, recommended[2]),
        #     (col4, recommended[3]),
        #     (col5, recommended[4])
        # )

        # for col in cols:
        #     movie = df_machine_learning[df_machine_learning["titre_str"] == col[1]]
        #     colonne = col[0]
        #     image_link = get_info(movie, "image")
        #     colonne.image(image_link, width = 135)
        #     movie_titre = get_info(movie, "titre_str")
        #     recommended_index.append(get_index_from_titre(df_site_web, movie_titre))
        #     test = colonne.button("clic", on_click = handle_click, key = movie_titre, args = recommended_index[])
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