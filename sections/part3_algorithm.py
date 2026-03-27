import streamlit as st
from utils.plots import (
    create_pca_step_by_step_plot, 
    create_uncentered_step_by_step_plot, 
    create_uncentered_pca_plot
)

def render_part3_algorithm():
    st.markdown("---")
    st.header("3. PCA Algorithm")
    st.write("### 3.1. Step-by-step algorithm execution scenario")

    st.markdown("""
    To produce the smooth projection results as guided in Part 1, the inner workings of tools like `sklearn.decomposition.PCA` must sequentially run through this 5-step lifecycle.

    1. **`Zero-center Data:`** The algorithm must shift the Center of Gravity of the data cloud (Mean - the Star coordinate) from being offset in space to anchoring exactly at the origin `(0, 0)`.
    2. **`Calculate Covariance Matrix C:`** Construct an ellipse representing the dispersion space of the data.
    3. **`Determine priority information dimensions:`** The most stretched out dispersion dimension of the ellipse is called Principal Component 1 (`PC1`). The narrower dimension holding less information is called `PC2`. They are always perpendicular to each other!
    4. **`Project Data:`** Because `PC1` retains the most variance, we project all the data solely onto that red axis. From the original coordinate dataset `[x,y]`, the information is excellently reduced to just a single coefficient `[x]`.

    ▶️ **Touch & Drag the progress bar below to see how the Data actually "moves"!**
    """)

    fig_pca_steps = create_pca_step_by_step_plot()
    st.plotly_chart(fig_pca_steps, use_container_width=True)

    st.markdown("---")
    st.write("### 3.2. Special case: What if it is NOT Zero-centered?")

    st.markdown("""
    What happens if we **FORGET STEP 1**, meaning we directly pass the original data matrix (far offset from `(0,0)`) into the Eigendecomposition analysis without shifting it back to the origin? The interactive feature below will clearly show you the disaster!
    """)

    fig_uncentered_steps = create_uncentered_step_by_step_plot()
    st.plotly_chart(fig_uncentered_steps, use_container_width=True)

    st.markdown("#### Why is the algorithm tricked into pointing in the direction of that Red arrow?")
    st.markdown("""
    Mathematically, PCA looks for a vector $v$ to maximize the projection variance:
    $$ \\arg\max_{v, ||v||=1} \\sum_{i=1}^{N} (v^T x_i)^2 $$

    Assume each data point $x_i$ is represented by the mean vector $m$ plus a small deviation $\\Delta x_i$ (i.e., $x_i = m + \\Delta x_i$). Then:
    $$ (v^T x_i)^2 = (v^T m + v^T \\Delta x_i)^2 $$

    If the data is **NOT** zero-centered (meaning the data cloud is located far from the origin), the magnitude of the mean vector $m$ will completely dominate the dispersion deviation $\\Delta x_i$ ($m \gg \\Delta x_i$). Then, the approximation expression becomes:
    $$ \\sum_{i=1}^N (v^T x_i)^2 \\approx N \\cdot (v^T m)^2 $$

    **💡 IMPORTANT CONCLUSION:**  
    For the term $(v^T m)^2$ to reach its maximum amplitude, the projection rule dictates that **Vector $v$ (the PC1 dimension) must lie parallel to the mean Vector $m$ itself**. 
    Because of this, the algorithm suffers from a severe "hallucination": It produces a solution where the PC1 vector points straight at the mean position of the cloud, instead of finding the stretched dimension of the data!
    """)

    st.markdown("""
    *➥ **Illustrated with a Vector Graph:** The chart below draws a few static rays (representing $x_i$) extending from the origin \([0,0]\) upwards. Using pure Vector addition algorithms, you can clearly see that the "Pseudo-PCA" Algorithm (red dashed line) has been completely tricked into following exactly 100% of the resultant direction of the Mean Vector (Yellow).*
    """)

    fig_uncentered_pca = create_uncentered_pca_plot()
    st.plotly_chart(fig_uncentered_pca, use_container_width=True)