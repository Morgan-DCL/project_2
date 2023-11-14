import streamlit as st
import pandas as pd

link = "machine_learning.parquet"

df_machine_learning = pd.read_parquet(link)

st.header("Movie Recommandation system")

movies = df_machine_learning["titre_str"]

# def get_movie_id(movie):



def get_image(movie_id):
    image = selected_movie["image"].iloc[0]
    return image

def get_video(movie_id):
    video_link = selected_movie["youtube"].iloc[0]
    return video_link + "?autoplay=1&autohide=2&border=0&wmode=opaque&enablejsapi=1&modestbranding=1&controls=0&showinfo=1&mute=1"

default_message = "Entrez ou sélectionnez le nom d'un film..."

selectvalue = st.selectbox("Select movie ⤵️", [default_message] + list(movies))

selected_movie = df_machine_learning[df_machine_learning["titre_str"] == selectvalue]

if selectvalue != default_message:
    st.button("Recommandations")

    image_link = get_image(selectvalue)
    st.image(image_link, caption = selectvalue, width = 300)
    video_link = get_video(selectvalue)
    # st.video(video_link, )
    
    video_html = """
                <video controls width="250" autoplay="true" muted="true" loop="true">
                <source 
                    src=/>
                /video>
                """


