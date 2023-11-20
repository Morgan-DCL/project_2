import sys
import os
sys.path.append(os.path.abspath(".."))
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotting import movies_by_decades

# Configuration de la page
st.set_page_config(
    page_title = "EDA Projet 2",
    page_icon = "📈",
    initial_sidebar_state = "collapsed"
)

link = "datasets/movies.parquet"
movies = pd.read_parquet(link)

st.title("Analyse de données exploratoires", anchor = False)
st.write("Nous avons importé tout les dataframes fourni pour le projet.")
st.write("Lors de l'importation on nettoie les données en changeant le type (float en int, etc...)")
st.write("On enlève les films à caractère pornographique avec la colonne 'isadult'")
st.write("On transforme les données dans les colonnes ayant plusieurs valeurs")
st.markdown("<br>", unsafe_allow_html = True)
st.write("ftessttesttestestetstest")
fig = movies_by_decades(movies)

fig[0]

fig[1]
# fig1 = go.Histogram(
#     x = movies["rating_avg"],
#     marker = dict(color="royalblue", line=dict(color="black", width=1)),
#     showlegend=False
# )
# median=movies["rating_avg"].median()
# max_=movies["rating_avg"].value_counts().max()
# fig1.add_shape(
#     go.layout.Shape(
#         type="line",
#         x0=median,
#         x1=median,
#         y0=0,
#         y1=1,
#         yref="paper",
#         line=dict(color="red", width=2, dash="dash"),
#     )
# )
# fig1.add_annotation(
#     x=median,
#     y=max_ + 100,
#     text=str(median),
#     name="Median",
#     showarrow=False,
#     xshift=15,
#     font=dict(color="red"),
# )
# fig1.add_trace(
#     go.Scatter(
#         x=[None],
#         y=[None],
#         mode="lines",
#         line=dict(color="red", width=2, dash="dash"),
#         name=f"Médiane",
#     )
# )
# fig1.update_layout(
#         title="Distribution des Notes Moyennes",
#         xaxis_title="Note Moyenne",
#         yaxis_title="Fréquence",
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             # y=0.99,
#             y=1.02,
#             xanchor="left",
#             # x=0.01
#             x=0.01,
#         ),
#     )
# fig1.show()