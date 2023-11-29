import os
import sys

sys.path.append(os.path.abspath(".."))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from plotting import movies_by_decades

# Configuration de la page
st.set_page_config(
    page_title="Persons Bio",
    page_icon="👤",
    initial_sidebar_state="collapsed",
    layout="wide",
)

if st.button("Retour"):
    switch_page("DDMRS")

index = st.session_state["actor"]
st.title(f"{index['name']}", anchor=False)