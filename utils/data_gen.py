import numpy as np
import pandas as pd
import streamlit as st

# 2. Data Generation Function
@st.cache_data
def generate_data(cov_matrix):
    np.random.seed(42) # For reproducibility
    mean = [0, 0]
    data = np.random.multivariate_normal(mean, cov_matrix, 100)
    return pd.DataFrame(data, columns=['Net Cash Flow', 'Net Worth Change'])

cov_matrix = [[5, 4], [4, 5]] # Creates a clear slanted ellipse

if 'data' not in st.session_state:
    st.session_state.data = generate_data(cov_matrix)

def regenerate_data():
    st.cache_data.clear()
    new_seed = np.random.randint(1000)
    np.random.seed(new_seed)
    mean = [0, 0]
    data = np.random.multivariate_normal(mean, cov_matrix, 100)
    st.session_state.data = pd.DataFrame(data, columns=['Net Cash Flow', 'Net Worth Change'])