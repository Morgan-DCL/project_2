import streamlit as st
import pandas as pd

machine_learning = "datasets/machine_learning_final.parquet"
site_web = "datasets/site_web.parquet"

df_machine_learning = pd.read_parquet(machine_learning)
df_site_web = pd.read_parquet(site_web)

movies = df_site_web["titre_str"]

def get_info(
        movie_id,
        info
    ):
    info = selected_movie[info].iloc[0]
    return info


st.header("Movie Recommandation system")
default_message = "Entrez ou sélectionnez le nom d'un film..."
selectvalue = st.selectbox(
    "Select movie ⤵️",
    [default_message] + list(sorted(movies))
)
selected_movie = df_site_web[df_site_web["titre_str"] == selectvalue]
if selectvalue != default_message:
    st.button("💡 Recommandations 💡")
    image_link = get_info(selectvalue, "image")
    col1, col2 = st.columns([1, 1])
    col1.image(image_link, width = 300)
    with col2:
        date = get_info(selectvalue, "date")
        st.header(f"{selectvalue} - ({date})")
        director_name = get_info(selectvalue, "director")
        actors_list = get_info(selectvalue, "actors")
        genre_list = get_info(selectvalue, "titre_genres")
        overview = get_info(selectvalue, "overview")
        st.markdown(
            f"**{genre_list}**",
            unsafe_allow_html=True
        )
        st.markdown(
            f"**Director:**<br/>{director_name}",
            unsafe_allow_html=True
        )
        st.markdown(
            f"**Actors :**<br/>{actors_list}",
            unsafe_allow_html=True
        )
        st.markdown(
            f"**Synopsis :**<br/>{overview}",
            unsafe_allow_html=True
        )
    video_link = get_info(
        selectvalue,
        "youtube"
    )
    st.video(video_link)