import streamlit as st
import numpy as np
from utils.plots import create_single_vector_projection_plot, create_interactive_covariance_plot

def render_part2_concepts():
    st.write("## 2. Fundamental Concepts")

    # 2.1 PROJECTION
    st.write("### 2.1 Concept of Projection")
    st.markdown("""
    To understand how PCA works, the physical theory of projection is essential. 
    - Assume we have a fixed vector $v$ (blue).
    - And a unit vector / directional line $u$ (red). 
    - When a light source is placed perpendicular to the axis $u$, the shadow of $v$ cast on the axis $u$ is the projection (green). 
        
    **Learning tool:** Try randomly rotating the axis $u$ using the slider below to see how vector $v$ "casts its shadow"!
    """)

    col_proj, _ = st.columns([2, 1])
    with col_proj:
        fig_single_proj = create_single_vector_projection_plot()
        st.plotly_chart(fig_single_proj, use_container_width=True)

    # 2.2 COVARIANCE MATRIX
    st.write("### 2.2 Covariance Matrix")
    st.markdown("""
    Our data cloud depends on the structure of the Covariance Matrix $\Sigma$ consisting of: the dispersion on the X-axis `Var(X)`, the dispersion on the Y-axis `Var(Y)`, and the positive/negative correlation `Cov(X, Y)`.

    **✨ SPECIAL FEATURE:** Select **1 parameter** (in the menu below) to turn it into a 60fps Animation slider, while keeping the other 2 variables fixed!
    """)

    anim_mode = st.radio("Select the parameter for smooth animation:", ["Cov(X, Y)", "Var(X)", "Var(Y)"], horizontal=True)

    scol1, scol2 = st.columns(2)
    if anim_mode == "Cov(X, Y)":
        with scol1: fixed_1 = st.slider("Fixed axis: Var(X)", 0.5, 10.0, 5.0, 0.5)
        with scol2: fixed_2 = st.slider("Fixed axis: Var(Y)", 0.5, 10.0, 5.0, 0.5)
        fig_sim = create_interactive_covariance_plot(anim_mode, fixed_1, fixed_2)
    elif anim_mode == "Var(X)":
        with scol1: fixed_1 = st.slider("Fixed axis: Var(Y)", 0.5, 10.0, 5.0, 0.5)
        max_c = np.sqrt(12.0 * fixed_1) - 0.1 
        with scol2: fixed_2 = st.slider("Fixed: Cov(X, Y)", float(-max_c), float(max_c), 0.0, 0.5)
        fig_sim = create_interactive_covariance_plot(anim_mode, fixed_1, fixed_2)
    else: # Var(Y)
        with scol1: fixed_1 = st.slider("Fixed axis: Var(X)", 0.5, 10.0, 5.0, 0.5)
        max_c = np.sqrt(12.0 * fixed_1) - 0.1
        with scol2: fixed_2 = st.slider("Fixed: Cov(X, Y)", float(-max_c), float(max_c), 0.0, 0.5)
        fig_sim = create_interactive_covariance_plot(anim_mode, fixed_1, fixed_2)

    st.plotly_chart(fig_sim, use_container_width=True)