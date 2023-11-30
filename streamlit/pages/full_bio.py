import os
import sys

sys.path.append(os.path.abspath(".."))

import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from datetime import datetime
from tools_app import (
    clean_dup,
    auto_scroll,
    get_clicked_bio,
    get_index_from_titre,
    infos_button,
)

df_sw = pd.read_parquet("datasets/site_web.parquet")
df_sw = clean_dup(df_sw)

# Configuration de la page
st.set_page_config(
    page_title="Persons Bio",
    page_icon="👤",
    initial_sidebar_state="collapsed",
    layout="wide",
)

st.session_state["clicked"] = None
st.session_state["clicked2"] = None
st.session_state["clicked3"] = None

if st.button("Retour"):
    switch_page("DDMRS")

hide_img_fs = """
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    </style>
"""
st.markdown(hide_img_fs, unsafe_allow_html=True)

round_corners = """
    <style>
        .st-emotion-cache-1v0mbdj > img{
            border-radius:2%;
        }
    </style>
"""
st.markdown(round_corners, unsafe_allow_html=True)

pdict = st.session_state["actor"]
mov_dup_dict : dict = st.session_state["dup_movie_dict"]

col1, col2 = st.columns([1, 4])
with col1:
    st.image(pdict["image"], use_column_width=True)
with col2:
    name, title = st.columns([1, 2])
    with name:
        st.header(
            f"{pdict['name']}", anchor=False, divider=True
        )
    birth = datetime.strptime(pdict['birthday'], '%Y-%m-%d') if pdict['birthday'] else "Unknow"
    end_date = datetime.strptime(pdict['deathday'], '%Y-%m-%d') if pdict['deathday'] else datetime.now()
    age = (end_date - birth).days // 365 if pdict['birthday'] else 0
    add_death = f" - {pdict['deathday']}" if pdict['deathday'] else ""

    st.caption(
        f"<p style='font-size: 16px;'>{birth.strftime('%Y-%m-%d') if pdict['birthday'] else 'Unknow'}{add_death} • ({age} ans)</p>",
        unsafe_allow_html=True
    )
    titre = "Réalisation" if pdict["director"] else "Célèbre pour"
    st.subheader(f"**{titre}**", anchor=False, divider=True)
    len_ml = len(pdict["top_5_movies_ids"])
    cols = st.columns(len_ml)

    for i, col in enumerate(cols):
        with col:
            nom_film, clicked3 = get_clicked_bio(
                pdict, mov_dup_dict, i, len_ml
            )
            if clicked3:
                st.session_state["clicked3"] = True
                infos_button(df_sw, st.session_state["movie_list"], get_index_from_titre(df_sw, nom_film))
    if st.session_state["clicked3"]:
        switch_page("DDMRS")
        # st.session_state["counter"] += 1
        # auto_scroll()
        # st.rerun()

st.subheader("**Biography :**", anchor=False, divider=True)
st.markdown(pdict['biography'])
