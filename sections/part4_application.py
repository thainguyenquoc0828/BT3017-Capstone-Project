import streamlit as st
import numpy as np
import os
from sklearn.decomposition import PCA

@st.cache_data(show_spinner=False)
def load_faces_and_compute_pca():
    # Use locally downloaded file instead of downloading via Internet
    # Ensure the file "olivetti_faces.npy" is in the root directory of your project (alongside app.py)
    local_path = "olivetti_faces.npy"
    
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File '{local_path}' not found. Please download the olivetti_faces.npy file and place it in the same directory as the application.")
        
    dataset = np.load(local_path)
    
    # The Olivetti dataset is saved in the shape of (400, 64, 64). 
    # We need to flatten it into an array X with shape (400, 4096)
    n_samples, h, w = dataset.shape
    X = dataset.reshape((n_samples, h * w))
    
    n_components = 150
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)
    
    components = pca.components_
    mean = pca.mean_
    explained_variance_ratio = pca.explained_variance_ratio_
    
    X_pca = pca.transform(X)
    
    return X, dataset, h, w, components, mean, explained_variance_ratio, X_pca

def render_part4_application():
    st.write("## 4. Practical Applications of PCA")
    st.markdown("""
    In this section, we will apply PCA to a real facial image dataset (Olivetti Faces dataset). 
    Image representation (which inherently has a high dimensionality - Width x Height) will be reduced to a few dozen or hundred PC dimensions.
    """)
    
    with st.spinner("Processing data from local file..."):
        try:
            X, images, h, w, components, mean, explained_variance_ratio, X_pca = load_faces_and_compute_pca()
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return
            
    st.markdown("---")
    st.write("### 4.1. Eigenfaces")
    st.markdown("""
    PCA helps extract the most essential "features" from an image dataset (called **Eigenfaces**). 
    - **Lighting / Shadow direction**
    - **Structure of jaw, eyes, nose**
    
    When we add the **Average Face** with varying intensities of these Eigenfaces (by adjusting the sliders), we can generate new faces.
    """)
    
    col_sliders, col_img = st.columns([1, 1])
    
    with col_sliders:
        st.write("**Adjust the weights of the first 5 Principal Components**")
        w1 = st.slider("PC 1 (e.g., Brightness)", -10.0, 10.0, 0.0, step=0.5)
        w2 = st.slider("PC 2", -10.0, 10.0, 0.0, step=0.5)
        w3 = st.slider("PC 3", -10.0, 10.0, 0.0, step=0.5)
        w4 = st.slider("PC 4", -10.0, 10.0, 0.0, step=0.5)
        w5 = st.slider("PC 5", -10.0, 10.0, 0.0, step=0.5)
        
    with col_img:
        st.write("**Reconstruction (Mean + Sum of PCs)**")
        scale_factor = np.sqrt(np.var(X_pca[:, 0:5], axis=0)) 
        
        custom_face_vector = mean + (
            (w1 * scale_factor[0] * components[0]) +
            (w2 * scale_factor[1] * components[1]) +
            (w3 * scale_factor[2] * components[2]) +
            (w4 * scale_factor[3] * components[3]) +
            (w5 * scale_factor[4] * components[4])
        )
        
        custom_face_img = custom_face_vector.reshape((h, w))
        custom_face_img = (custom_face_img - custom_face_img.min()) / (custom_face_img.max() - custom_face_img.min() + 1e-8)
        
        st.image(custom_face_img, width=300, caption="Linearly combined face")


    st.markdown("---")
    st.write("### 4.2. Image Compression using PCA")
    st.markdown(f"""
    Representing an image (from the Olivetti dataset) requires data for all pixels: $64 \\times 64 = 4096$ dimensions. 
    However, we only need $K$ PCA dimensions to reconstruct the image with almost original detail. 
    Drag the slider $K$ below to see the change!
    """)
    
    if 'test_img_idx' not in st.session_state:
        st.session_state.test_img_idx = 0
    
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("Change to random image"):
            st.session_state.test_img_idx = np.random.randint(0, len(X))
            
    img_idx = st.session_state.test_img_idx
    original_face = X[img_idx]
    
    k = st.slider("Number of PCA dimensions (K PCs) retained", min_value=1, max_value=150, value=20, step=1)
    
    k_components_weights = X_pca[img_idx, :k]
    reconstructed_face = mean + np.dot(k_components_weights, components[:k, :])
    
    col_orig, col_recon = st.columns(2)
    with col_orig:
        orig_img = original_face.reshape((h, w))
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
        st.image(orig_img, width=300, caption="Original image")
        
    with col_recon:
        recon_img = reconstructed_face.reshape((h, w))
        recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)
        st.image(recon_img, width=300, caption=f"Compressed and reconstructed image (K={k})")
        
    retained_variance = np.sum(explained_variance_ratio[:k]) * 100
    st.success(f"**Amount of information (Variance) retained:** {retained_variance:.2f}% with **{k}** dimensions (Reduced from {h*w} to {k}).")