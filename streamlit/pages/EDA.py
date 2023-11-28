import sys
import os
sys.path.append(os.path.abspath(".."))
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plot import (
    movies_by_decades,
    movies_duration_by_decades_boxplot,
    movies_top_x,
    actors_top_1_by_decades,
    actors_top_10_by_genres,
    actors_top_by_movies,
    actors_top_10_by_votes,
)

# Configuration de la page
st.set_page_config(
    page_title = "EDA Projet 2",
    page_icon = "📈",
    initial_sidebar_state = "collapsed"
)

link = "datasets/movies.parquet"
movies = pd.read_parquet(link)

st.title("Analyse de données exploratoires", anchor = False)
st.markdown("<br>", unsafe_allow_html = True)
st.write("Retrait des films pour adultes, des films pas encore sortis et des films n'ayant pas été adapté en français")
col1, col2 = st.columns(2)
with col1:
    st.write("Etude démographique de la Creuse")
    st.image("https://image.noelshack.com/fichiers/2023/48/1/1701096598-etudedemographique.png")
with col2:
    st.markdown("""<a href="https://fr.statista.com/statistiques/498200/preference-films-etrangers-vo-vf-france/" rel="nofollow"><img src="https://fr.statista.com/graphique/1/498200/preference-films-etrangers-vo-vf-france.jpg" alt="Statistique: Préférez-vous plutôt voir les films étrangers en version française ou en version originale sous-titrée ? | Statista" style="width: 100%; height: auto !important; max-width:1000px;-ms-interpolation-mode: bicubic;"/></a>""", unsafe_allow_html=True)
st.write("Analyse des films présents dans le dataset après premier nettoyage")
fig = movies_by_decades(movies)
col1, col2, col3 = st.columns(3)
with col2:
    fig[0]
col4, col5 = st.columns(2)
with col4:
    fig[1]
    fig[5]
with col5:
    fig[2]
    fig[3]


link2 = "datasets/movies_cleaned.parquet"
movies_cleaned = pd.read_parquet(link2)

st.write("Retrait des films sorti avant 1960, des films ayant une note inférieure à 6.3 et ceux ayant reçus moins de 5004 votes.")

figs = movies_by_decades(movies_cleaned)
col1, col2, col3 = st.columns(3)
with col2:
    figs[0]
col4, col5 = st.columns(2)
with col4:
    figs[1]
    figs[5]
with col5:
    figs[2]
    figs[3]


duration = movies_duration_by_decades_boxplot(movies_cleaned)
col1, col2, col3 = st.columns(3)
with col2:
    duration

movies_top_10 = movies_top_x(movies_cleaned, 10)
col1, col2, col3 = st.columns(3)
with col2: movies_top_10

link = "datasets/actors_movies.parquet"
actors = pd.read_parquet(link)

top1_decades = actors_top_1_by_decades(actors)
top_by_movies = actors_top_by_movies(actors, 10)
top_by_genres = actors_top_10_by_genres(actors, 10)
top_by_votes = actors_top_10_by_votes(actors, 10)

col1, col2 = st.columns(2)
with col1:
    top1_decades
    top_by_genres

with col2:
    top_by_movies
    top_by_votes

