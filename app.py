import streamlit as st
import numpy as np
import pandas as pd

# ---> IMPORT TỪ THƯ MỤC ƯTILS Ở ĐÂY <---
from utils.data_gen import generate_data, regenerate_data
from utils.plots import (
    create_projection_plot, create_interactive_plotly,
    create_single_vector_projection_plot, create_interactive_covariance_plot,
    create_pca_step_by_step_plot, create_uncentered_pca_plot,
    create_uncentered_step_by_step_plot
)

from sections.part1_intro import render_part1_intro
from sections.part2_concepts import render_part2_concepts
from sections.part3_algorithm import render_part3_algorithm
from sections.part4_application import render_part4_application

# 1. Page Configuration
st.set_page_config(page_title="Principal Component Analysis (PCA) learning tools and visualization", layout="wide")

# --- CUSTOM CSS FOR DARK THEME ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #121212;
        color: white;
    }
    [data-testid="stHeader"] {
        background-color: #121212;
    }
    div.stButton > button:first-child {
        background-color: #333;
        color: white;
        border: 1px solid #555;
    }
</style>
""", unsafe_allow_html=True)

if 'data' not in st.session_state:
    cov_matrix = [[5, 4], [4, 5]]
    st.session_state.data = generate_data(cov_matrix)

# 3. App Title
st.title("Principal Component Analysis (PCA) learning tools and visualization")

# 4. TRÌNH BÀY CÁC PHẦN DƯỚI DẠNG SINGLE-PAGE
render_part1_intro()
render_part2_concepts()
render_part3_algorithm()
render_part4_application()
