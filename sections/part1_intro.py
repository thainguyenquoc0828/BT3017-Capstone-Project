import streamlit as st
import numpy as np
from utils.plots import create_projection_plot, create_interactive_plotly

def render_part1_intro():
    st.write("## 1. Introduction")

    # Data processing from session_state
    df = st.session_state.data
    X = df.values
    X = X - np.mean(X, axis=0) 

    cov_matrix_empirical = np.cov(X.T)
    total_variance = np.trace(cov_matrix_empirical)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix_empirical)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    pc1_vector = eigenvectors[:, sorted_indices].T[0]
    pc1_variance = eigenvalues[sorted_indices][0]
    pc2_vector = eigenvectors[:, sorted_indices].T[1]
    pc2_variance = eigenvalues[sorted_indices][1]

    st.markdown("""
    Imagine you are a bank employee. Suppose your boss has a dataset of clients, where each client has **2 different financial features**:

    - **Net Cash Flow** (Monthly surplus / deficit)
    - **Net Worth Change** (Increase / decrease in assets)

    and asks you to summarize these clients into **1 feature** (a single number). 

    **Question:** Which dimension should we choose to project the data onto without losing too much information?
    """)

    st.markdown("---")
    st.write("### How is information defined here?")
    st.markdown("""
    In the context of dimensionality reduction, we define **Information = Data spread (Variance)**.
    - If we project the data onto a 1D line and the points are highly dispersed, we will **retain a lot of information**.
    - Conversely, if the points overlap each other, we will **lose a lot of information**.
    To illustrate, let's look at the two examples of two projections here, can you tell which plot retains the most information? 
    """)

    col_good, col_bad = st.columns(2)
    with col_good:
        fig_good = create_projection_plot(X, pc1_vector, pc1_variance, total_variance, "Example 1")
        st.plotly_chart(fig_good, use_container_width=True)

    with col_bad:
        fig_bad = create_projection_plot(X, pc2_vector, pc2_variance, total_variance, "Example 2")
        st.plotly_chart(fig_bad, use_container_width=True)

    st.markdown("""
    - We can see in the left chart that the projected points are highly dispersed, reflecting that a lot of information will be retained if projected onto this axis.
    - Conversely, in the right chart, many projected points overlap. If we project here, we won't retain much information.
    """)

    st.markdown("---")
    st.write("### Experiment with projecting onto X and Y axis")
    st.markdown("""
    We will try projecting the data onto each axis:
    - Net Cash Flow
    - Net Worth Change
    """)

    col_x, col_y = st.columns(2)
    vector_x = np.array([1, 0])
    variance_x = cov_matrix_empirical[0, 0]
    with col_x:
        fig_x = create_projection_plot(X, vector_x, variance_x, total_variance, "Projection onto X axis")
        st.plotly_chart(fig_x, use_container_width=True)

    vector_y = np.array([0, 1])
    variance_y = cov_matrix_empirical[1, 1]
    with col_y:
        fig_y = create_projection_plot(X, vector_y, variance_y, total_variance, "Projection onto Y axis")
        st.plotly_chart(fig_y, use_container_width=True)

    st.markdown("""
    - When projecting onto individual axis, **about 50%** of the information is retained, meaning many points will **overlap with the projections of other points**.
    - However, we want as few overlapping points as possible. Therefore, try thinking about how to rotate the axis.
    """)

    st.markdown("---")
    st.write("### Changing angles: Which axis is the most optimal?")
    st.markdown("""
    Try dragging the slider below yourself to rotate the line in 2D space, and pay attention to the hint at the top. Try to find the line that contains the most information!
    """)

    fig_interactive = create_interactive_plotly(X, total_variance, pc1_vector)
    st.plotly_chart(fig_interactive, use_container_width=True)

    st.markdown("""
    > **Conclusion:** When you adjust the slider until the system says **CORRECT**, that is the angle where the maximum amount of information (variance) is retained. 
    > In short: Instead of tiring your hands dragging the slider, **PCA** is the mathematical tool that helps you **automatically** pinpoint the exact best angle (whether the data is 2D, 3D, or 100D)!
    """)