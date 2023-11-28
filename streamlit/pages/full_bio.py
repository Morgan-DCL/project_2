import os
import sys

sys.path.append(os.path.abspath(".."))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from plotting import movies_by_decades

# Configuration de la page
st.set_page_config(
    page_title="Persons Bio",
    page_icon="👤",
    initial_sidebar_state="collapsed",
    layout="wide",
)

st.title("Analyse de données exploratoires", anchor=False)
st.write("Nous avons importé tout les dataframes fourni pour le projet.")
st.write(
    "Lors de l'importation on nettoie les données en changeant le type (float en int, etc...)"
)
st.write(
    "On enlève les films à caractère pornographique avec la colonne 'isadult'"
)
st.write(
    "On transforme les données dans les colonnes ayant plusieurs valeurs"
)
st.markdown("<br>", unsafe_allow_html=True)
st.write("ftessttesttestestetstest")

st.markdown(
    '<a href="/next_page" target="_self">Next page</a>',
    unsafe_allow_html=True,
)
