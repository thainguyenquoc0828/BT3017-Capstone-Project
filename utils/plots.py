import numpy as np
import plotly.graph_objects as go
from scipy.stats import chi2

# ==========================================
# COMMON FUNCTION: Static Plot (Section 1)
# ==========================================
def create_projection_plot(X, vector, variance, total_variance, title):
    vector = vector / np.linalg.norm(vector)
    projections = X @ vector.T
    proj_points = projections.reshape(-1, 1) * vector
    
    line_range = 10
    axis_points = np.array([[-line_range * vector[0], line_range * vector[0]],
                             [-line_range * vector[1], line_range * vector[1]]])

    line_trace_x = []
    line_trace_y = []
    for i in range(len(X)):
        line_trace_x.extend([X[i, 0], proj_points[i, 0], None])
        line_trace_y.extend([X[i, 1], proj_points[i, 1], None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', name='Data Points', marker=dict(color='#007bff', opacity=0.6, size=6)))
    fig.add_trace(go.Scatter(x=axis_points[0], y=axis_points[1], mode='lines', name='Projection Axis', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=proj_points[:, 0], y=proj_points[:, 1], mode='markers', name='Projected Points', marker=dict(color='#28a745', size=8, symbol='x')))
    fig.add_trace(go.Scatter(x=line_trace_x, y=line_trace_y, mode='lines', line=dict(color='#aaaaaa', dash='dash', width=1), showlegend=False))

    pct_retained = (variance / total_variance) * 100
    stats_text = f"<b>VARIANCE:</b> {variance:.3f}<br><b>INFORMATION RETAINED:</b> {pct_retained:.1f}%"

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=18)),
        xaxis_title="Net Cash Flow (X)", xaxis_title_font_color="white",
        yaxis_title="Net Worth Change (Y)", yaxis_title_font_color="white",
        xaxis=dict(range=[-8, 8], constrain='domain', tickfont_color="white", gridcolor="#333"),
        yaxis=dict(range=[-8, 8], scaleanchor="x", scaleratio=1, tickfont_color="white", gridcolor="#333"),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font_color="white", bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[dict(xref="paper", yref="paper", x=0.99, y=0.99, showarrow=False, text=stats_text, align="right", font=dict(color="white", size=16))]
    )
    return fig

# ==========================================
# ADVANCED FUNCTION: Real-time Plotly Animation for Section 1
# ==========================================
def create_interactive_plotly(X, total_variance, pc1_vector):
    opt_rad = np.arctan2(pc1_vector[1], pc1_vector[0])
    opt_a = int(round(np.degrees(opt_rad))) % 180
    worst_a = (opt_a + 90) % 180

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', name='Data Points', marker=dict(color='#007bff', opacity=0.6, size=6)))
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Projection Axis', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Projected Points', marker=dict(color='#28a745', size=8, symbol='x')))
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Dashed Lines', line=dict(color='#aaaaaa', dash='dash', width=1), showlegend=False))
    fig.add_trace(go.Scatter(x=[0], y=[9.2], mode="text", name='Status_Hint', text=[""], textposition="bottom center", textfont=dict(size=20, weight="bold"), showlegend=False))
    fig.add_trace(go.Scatter(x=[0], y=[-9.2], mode="text", name='Status_Stats', text=[""], textposition="top center", textfont=dict(size=16, color="white"), showlegend=False))

    frames = []
    slider_steps = []

    for a in range(181):
        theta = np.radians(a)
        v = np.array([np.cos(theta), np.sin(theta)])
        proj = X @ v.T
        proj_pts = proj.reshape(-1, 1) * v
        
        lt_x, lt_y = [], []
        for i in range(len(X)):
            lt_x.extend([X[i, 0], proj_pts[i, 0], None])
            lt_y.extend([X[i, 1], proj_pts[i, 1], None])
            
        variance = np.var(proj)
        pct = (variance / total_variance) * 100
        
        diff_best = min((a - opt_a) % 180, (opt_a - a) % 180)
        diff_worst = min((a - worst_a) % 180, (worst_a - a) % 180)
        
        if diff_best == 0:
            msg, color = f"🎉 EXACT! You found the optimal angle ({opt_a}°)", "#00ff00"
        elif diff_best <= 5:
            msg, color = "🔥 VERY CLOSE! The axis you need is right around here...", "#ffaa00"
        elif diff_best <= 15:
            msg, color = "👍 Right direction! Keep turning slowly", "#ffff00"
        elif diff_worst == 0:
            msg, color = f"💀 WORST! This is the angle that LOSES THE MOST information ({worst_a}°)", "#ff0000"
        elif diff_worst <= 15:
            msg, color = "📉 Warning: Almost all information is lost, wrong way!", "#ff4d4d"
        else:
            msg, color = "🧊 Information is changing, keep turning!", "#aaaaaa"
            
        stats_text = f"<b>RETAINED VARIANCE:</b> {variance:.3f} | {pct:.1f}% INFORMATION"
        
        frame = go.Frame(
            name=str(a),
            data=[
                go.Scatter(x=X[:, 0], y=X[:, 1]),
                go.Scatter(x=[-10*v[0], 10*v[0]], y=[-10*v[1], 10*v[1]]),
                go.Scatter(x=proj_pts[:, 0], y=proj_pts[:, 1]),
                go.Scatter(x=lt_x, y=lt_y),
                go.Scatter(x=[0], y=[9.2], text=[msg], textfont=dict(color=color)),
                go.Scatter(x=[0], y=[-9.2], text=[stats_text])
            ]
        )
        frames.append(frame)
        
        step = dict(
            method="animate",
            args=[[str(a)], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))],
            label=str(a)
        )
        slider_steps.append(step)

    fig.frames = frames
    init_a = 0
    fig.update_traces(x=frames[init_a].data[1].x, y=frames[init_a].data[1].y, selector=dict(name='Projection Axis'))
    fig.update_traces(x=frames[init_a].data[2].x, y=frames[init_a].data[2].y, selector=dict(name='Projected Points'))
    fig.update_traces(x=frames[init_a].data[3].x, y=frames[init_a].data[3].y, selector=dict(name='Dashed Lines'))
    fig.update_traces(text=frames[init_a].data[4].text, textfont=frames[init_a].data[4].textfont, selector=dict(name='Status_Hint'))
    fig.update_traces(text=frames[init_a].data[5].text, selector=dict(name='Status_Stats'))

    fig.update_layout(
        xaxis_title="Net Cash Flow (X)", xaxis=dict(range=[-10, 10], constrain='domain', gridcolor="#333", tickfont_color="white"),
        yaxis_title="Net Worth Change (Y)", yaxis=dict(range=[-10, 10], scaleanchor="x", scaleratio=1, gridcolor="#333", tickfont_color="white"),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False, height=650,
        sliders=[dict(
            active=init_a, font=dict(color="white"), bgcolor="#444", bordercolor="#666",
            currentvalue={"prefix": "Test angle: ", "suffix": "°", "font": dict(color="white", size=18)},
            pad={"t": 30}, steps=slider_steps
        )], updatemenus=[]
    )
    return fig

# ==========================================
# ADVANCED FUNCTION: Vector Projection (Learning Tool)
# ==========================================
def create_single_vector_projection_plot():
    v = np.array([4, 3])
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, v[0]], y=[0, v[1]], mode='lines+markers+text', name='Vector v', 
        line=dict(color='#007bff', width=4), marker=dict(size=8),
        text=["", f"<b>v</b> [{v[0]}, {v[1]}]"], textposition="top center", textfont=dict(color="#007bff", size=16)
    ))
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+text', name='Projection Axis u', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers+text', name='Projection', line=dict(color='#28a745', width=6)))
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Perpendicular line', line=dict(color='white', dash='dash', width=2)))
    
    # Trace for Info Dashboard
    fig.add_trace(go.Scatter(x=[-5.5], y=[5.5], mode="text", name="Dashboard", text=[""], textposition="bottom right", textfont=dict(size=14, color="white", family="monospace"), showlegend=False))

    frames = []
    slider_steps = []

    for a in range(361): 
        theta = np.radians(a)
        u = np.array([np.cos(theta), np.sin(theta)]) 
        
        dot_product = np.dot(v, u)
        proj = dot_product * u
        proj_length = abs(dot_product)
        
        dashboard_text = (
            f"<b>REAL-TIME PARAMS:</b><br><br>"
            f"<b>Fixed Vector v:</b> [ {v[0]:.2f} , {v[1]:.2f} ]<br>"
            f"<b>Projection Axis u:</b> [ {u[0]:.2f} , {u[1]:.2f} ]<br>"
            f"<span style='color:#28a745'><b>Projection length (||p||):</b> {proj_length:.2f}</span>"
        )
        
        frame = go.Frame(
            name=str(a),
            data=[
                go.Scatter(x=[0, v[0]], y=[0, v[1]]),
                go.Scatter(
                    x=[-10*u[0], 10*u[0]], y=[-10*u[1], 10*u[1]],
                    text=["", f"<b>u</b> [{u[0]:.2f}, {u[1]:.2f}]"],
                    textposition="bottom right", textfont=dict(color="red", size=14)
                ),
                go.Scatter(
                    x=[0, proj[0]], y=[0, proj[1]],
                    text=["", f"<b>||p||</b> = {proj_length:.2f}"],
                    textposition="bottom right", textfont=dict(color="#28a745", size=14)
                ),
                go.Scatter(x=[v[0], proj[0]], y=[v[1], proj[1]]),
                go.Scatter(x=[-5.5], y=[5.5], text=[dashboard_text])
            ]
        )
        frames.append(frame)
        
        step = dict(
            method="animate",
            args=[[str(a)], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))],
            label=str(a)
        )
        slider_steps.append(step)

    fig.frames = frames
    init_a = 45 
    fig.update_traces(x=frames[init_a].data[1].x, y=frames[init_a].data[1].y, text=frames[init_a].data[1].text, textposition=frames[init_a].data[1].textposition, textfont=frames[init_a].data[1].textfont, selector=dict(name='Projection Axis u'))
    fig.update_traces(x=frames[init_a].data[2].x, y=frames[init_a].data[2].y, text=frames[init_a].data[2].text, textposition=frames[init_a].data[2].textposition, textfont=frames[init_a].data[2].textfont, selector=dict(name='Projection'))
    fig.update_traces(x=frames[init_a].data[3].x, y=frames[init_a].data[3].y, selector=dict(name='Perpendicular line'))
    fig.update_traces(text=frames[init_a].data[4].text, selector=dict(name='Dashboard'))

    fig.update_layout(
        xaxis=dict(range=[-6, 6], constrain='domain', gridcolor="#333", tickfont_color="white"),
        yaxis=dict(range=[-6, 6], scaleanchor="x", scaleratio=1, gridcolor="#333", tickfont_color="white"),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        margin=dict(l=20, r=20, t=10, b=20),
        showlegend=True, legend=dict(font_color="white", bgcolor='rgba(0,0,0,0)'),
        height=650,
        sliders=[dict(
            active=init_a, font=dict(color="white"), bgcolor="#444", bordercolor="#666",
            currentvalue={"prefix": "Axis u angle: ", "suffix": "°", "font": dict(color="white", size=16)},
            pad={"t": 20}, steps=slider_steps
        )], updatemenus=[]
    )
    return fig

# ==========================================
# ADVANCED FUNCTION: Interactive Covariance Matrix (Multi-Mode)
# ==========================================
def create_interactive_covariance_plot(anim_mode, fixed_1, fixed_2):
    fig = go.Figure()
    
    np.random.seed(42)
    base_data = np.random.randn(300, 2)
    
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(color='rgba(23, 162, 184, 0.6)', size=7), name="Data"))
    fig.add_trace(go.Scatter(x=[-11.5], y=[11.5], mode="text", name="Dashboard", text=[""], textposition="bottom right", textfont=dict(size=14, color="white", family="monospace"), showlegend=False))

    frames = []
    slider_steps = []
    prefix_txt = ""

    # Determine accurate Mathematical limit based on Positive Semi-Definite formula
    if anim_mode == "Cov(X, Y)":
        var_x, var_y = fixed_1, fixed_2
        max_cov_limit = np.sqrt(var_x * var_y) - 0.05
        anim_values = np.linspace(-max_cov_limit, max_cov_limit, 81)
        prefix_txt = "Smooth animate Cov(X,Y): "
    elif anim_mode == "Var(X)":
        var_y, cov = fixed_1, fixed_2
        min_v = (cov**2 / var_y) + 0.05 if var_y != 0 else 0.1
        anim_values = np.linspace(max(0.1, min_v), 12.0, 81)
        prefix_txt = "Smooth animate Var(X): "
    else: # Var(Y)
        var_x, cov = fixed_1, fixed_2
        min_v = (cov**2 / var_x) + 0.05 if var_x != 0 else 0.1
        anim_values = np.linspace(max(0.1, min_v), 12.0, 81)
        prefix_txt = "Smooth animate Var(Y): "
        
    for i, val in enumerate(anim_values):
        if anim_mode == "Cov(X, Y)":
            cov = val
        elif anim_mode == "Var(X)":
            var_x = val
        else:
            var_y = val
            
        cov_matrix = np.array([[var_x, cov], 
                               [cov, var_y]])
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(cov_matrix)
            transformed_data = np.dot(base_data, L.T)
        except np.linalg.LinAlgError:
            transformed_data = np.zeros_like(base_data) # Safe fallback
        
        latex_str = (f"<b>COVARIANCE MATRIX:</b><br><br>"
                     f"<b>Var(X):</b> <span style='color:{'#00e5ff' if anim_mode=='Var(X)' else 'white'}'>{var_x:.2f}</span> | "
                     f"<b>Var(Y):</b> <span style='color:{'#00e5ff' if anim_mode=='Var(Y)' else 'white'}'>{var_y:.2f}</span><br>"
                     f"<b>Cov(X, Y):</b> <span style='color:{'#00e5ff' if anim_mode=='Cov(X, Y)' else 'white'}'>{cov:.2f}</span><br>"
                     f"Cov Meaning: {'Positively correlated' if cov > 0 else 'Negatively correlated' if cov < 0 else 'Uncorrelated'}")
        
        frame = go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=transformed_data[:, 0], y=transformed_data[:, 1]),
                go.Scatter(x=[-11.5], y=[11.5], text=[latex_str])
            ]
        )
        frames.append(frame)
        
        step = dict(
            method="animate",
            args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))],
            label=f"{val:.1f}"
        )
        slider_steps.append(step)
        
    fig.frames = frames
    
    init_i = len(anim_values) // 2 
    fig.update_traces(x=frames[init_i].data[0].x, y=frames[init_i].data[0].y, selector=dict(name='Data'))
    fig.update_traces(text=frames[init_i].data[1].text, selector=dict(name='Dashboard'))

    fig.update_layout(
        xaxis_title="Net Cash Flow (X)", yaxis_title="Net Worth Change (Y)",
        xaxis=dict(range=[-12, 12], constrain='domain', gridcolor="#333", tickfont_color="white"),
        yaxis=dict(range=[-12, 12], scaleanchor="x", scaleratio=1, gridcolor="#333", tickfont_color="white"),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        margin=dict(l=20, r=20, t=10, b=20), height=550, showlegend=False,
        sliders=[dict(
            active=init_i, font=dict(color="white"), bgcolor="#444", bordercolor="#666",
            currentvalue={"prefix": prefix_txt, "font": dict(color="#00e5ff", size=16)},
            pad={"t": 20}, steps=slider_steps
        )]
    )
    return fig

# ==========================================
# ADVANCED FUNCTION: Step-by-Step PCA Algorithm (Section 3.1)
# ==========================================
def create_pca_step_by_step_plot():
    np.random.seed(42)
    # 1. Original Data (uncentered - shifted to [8, 8])
    mean_orig = np.array([8, 8])
    cov_orig = np.array([[7, 5], [5, 6]])
    X_raw = np.random.multivariate_normal(mean_orig, cov_orig, 100)
    
    # 2. Shift to origin (0,0) (Zero-center)
    mean_X = np.mean(X_raw, axis=0)
    X_centered = X_raw - mean_X
    
    # 3. Calculate Covariance matrix and Eigenvectors
    cov_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate coordinates for bounding Ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    ellipse = (transform @ circle) * np.sqrt(chi2.ppf(0.95, 2))
    
    # Standard length of Eigen Vector
    scale = 3.0
    pc1 = eigenvectors[:, 0] * np.sqrt(eigenvalues[0]) * scale
    pc2 = eigenvectors[:, 1] * np.sqrt(eigenvalues[1]) * scale
    
    axis_len = 25
    pc1_axis = eigenvectors[:, 0] * axis_len
    
    # 4. Projection
    v1 = eigenvectors[:, 0]
    proj_pts = (X_centered @ v1).reshape(-1, 1) * v1
    
    lt_x, lt_y = [], []
    for i in range(len(X_centered)):
        lt_x.extend([X_centered[i, 0], proj_pts[i, 0], None])
        lt_y.extend([X_centered[i, 1], proj_pts[i, 1], None])

    fig = go.Figure()

    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.2)", layer='below')
    fig.add_vline(x=0, line_dash="solid", line_color="rgba(255,255,255,0.2)", layer='below')

    # INITIALIZE WITH X,Y = [None] INSTEAD OF NOTHING SO ANIMATION CAN RESET
    null_value = [None]
    fig.add_trace(go.Scatter(x=X_raw[:,0], y=X_raw[:,1], mode='markers', name='Data', marker=dict(color='#007bff', size=8, opacity=0.8))) 
    fig.add_trace(go.Scatter(x=[mean_X[0]], y=[mean_X[1]], mode='markers', name='Mean Point', marker=dict(color='yellow', size=14, symbol='star'))) 
    fig.add_trace(go.Scatter(x=null_value, y=null_value, mode='lines', name='Distribution Ellipse', line=dict(color='#E1BEE7', dash='dash', width=2))) 
    fig.add_trace(go.Scatter(x=null_value, y=null_value, mode='lines+markers+text', name='PC1 (Primary)', line=dict(color='red', width=5), marker=dict(size=10, symbol='arrow-bar-up', angleref='previous'), textfont=dict(color="#ff3333", size=18, family="Arial Black"), textposition="top right")) 
    fig.add_trace(go.Scatter(x=null_value, y=null_value, mode='lines+markers+text', name='PC2 (Secondary)', line=dict(color='green', width=4), marker=dict(size=10, symbol='arrow-bar-down', angleref='previous'), textfont=dict(color="#33ff33", size=18, family="Arial Black"), textposition="bottom right")) 
    fig.add_trace(go.Scatter(x=null_value, y=null_value, mode='lines', name='Projection line', line=dict(color='#aaaaaa', dash='dash', width=1), showlegend=False)) 
    
    fig.add_trace(go.Scatter(x=[3], y=[22], mode="text", name='Instructions', text=["<b>STEP 1:</b> Initial data is usually offset from the origin."], textfont=dict(size=18, color="#00e5ff"), textposition="top center", showlegend=False))

    frames = []
    
    # Clear variables for empty text and coordinates
    null_t = ["", ""]

    # Matrix: data, x_mean, y_mean, x_elips, y_elips, x_pc1, y_pc1, x_pc2, y_pc2, x_dash, y_dash, txt_pc1, txt_pc2, text_HD
    steps_data = [
        ("Original", X_raw, [mean_X[0]], [mean_X[1]], 
         null_value, null_value, null_value, null_value, null_value, null_value, null_value, null_value, null_t, null_t, "<b>STEP 1: Original Data</b> - Data is far shifted to the top corner."),
        
        ("Zero-Center", X_centered, [0], [0], 
         null_value, null_value, null_value, null_value, null_value, null_value, null_value, null_value, null_t, null_t, "<b>STEP 2: Zero-Center</b> - Subtract the Mean to pull the Center of Gravity straight to (0,0)."),
        
        ("Covariance", X_centered, [0], [0], 
         ellipse[0,:], ellipse[1,:], null_value, null_value, null_value, null_value, null_value, null_value, null_t, null_t, "<b>STEP 3: Covariance Matrix (C)</b> - Describe distribution with an ellipse."),
        
        ("PCA Vectors", X_centered, [0], [0], 
         ellipse[0,:], ellipse[1,:], [0, pc1[0]], [0, pc1[1]], [0, pc2[0]], [0, pc2[1]], null_value, null_value, ["", "<b>PC1</b>"], ["", "<b>PC2</b>"], "<b>STEP 4: Eigen PC1 & PC2</b> - Algorithm core: identify 2 variance vectors."),
         
        ("Projection", proj_pts, [0], [0], 
         null_value, null_value, [-pc1_axis[0], pc1_axis[0]], [-pc1_axis[1], pc1_axis[1]], null_value, null_value, lt_x, lt_y, null_t, null_t, "<b>STEP 5: Projection</b> - Remove ellipse & PC2, project data solely onto PC1.")
    ]

    slider_steps = []
    for i, (name, x_d, xm, ym, xe, ye, xpc1, ypc1, xpc2, ypc2, xdash, ydash, tpc1, tpc2, txt) in enumerate(steps_data):
        frame = go.Frame(
            name=f"step_{i}",
            data=[
                go.Scatter(x=x_d[:,0], y=x_d[:,1]),
                go.Scatter(x=xm, y=ym),
                go.Scatter(x=xe, y=ye),
                go.Scatter(x=xpc1, y=ypc1, text=tpc1), # Enable Text PC1
                go.Scatter(x=xpc2, y=ypc2, text=tpc2), # Enable Text PC2
                go.Scatter(x=xdash, y=ydash),
                go.Scatter(x=[3], y=[22], text=[txt])
            ]
        )
        frames.append(frame)
        
        slider_steps.append(dict(
            method="animate",
            args=[[f"step_{i}"], dict(mode="immediate", frame=dict(duration=1200, redraw=False), transition=dict(duration=1200, easing="cubic-in-out"))],
            label=f"Step {i+1}"
        ))

    fig.frames = frames

    fig.update_layout(
        xaxis=dict(range=[-18, 25], constrain='domain', gridcolor="#333", tickfont_color="white"),
        yaxis=dict(range=[-18, 25], scaleanchor="x", scaleratio=1, gridcolor="#333", tickfont_color="white"),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        margin=dict(l=20, r=20, t=20, b=20),
        height=680, legend=dict(font_color="white", bgcolor='rgba(0,0,0,0)', x=0.01, y=0.95),
        sliders=[dict(
            active=0, font=dict(color="white"), bgcolor="#444", bordercolor="#666",
            currentvalue={"prefix": "Algorithm Progress: ", "font": dict(color="white", size=16)},
            pad={"t": 30}, steps=slider_steps
        )], updatemenus=[]
    )
    return fig

# ==========================================
# ADVANCED FUNCTION: Uncentered PCA (Section 3.2)
# ==========================================
def create_uncentered_pca_plot():
    np.random.seed(42)
    # Repeat the same skewed elongated shape of the above function
    cov_skew = np.array([[8.0, -6.5], [-6.5, 6.0]])
    center_shift = np.array([12, 12])
    X = np.random.multivariate_normal(center_shift, cov_skew, 150)
    mean_X = np.mean(X, axis=0)
    
    # Calculate True Vector
    cov_true = np.cov(X.T)
    evals_true, evecs_true = np.linalg.eig(cov_true)
    idx_t = evals_true.argsort()[::-1]
    pc1_true = evecs_true[:, idx_t[0]]

    # Run "Pseudo-PCA" 
    pseudo_cov = (X.T @ X) / (X.shape[0] - 1)
    eigenvalues_nc, eigenvectors_nc = np.linalg.eig(pseudo_cov)
    idx_nc = eigenvalues_nc.argsort()[::-1]
    pc1_nc = eigenvectors_nc[:, idx_nc[0]]
    
    fig = go.Figure()
    
    # 1. Draw faint vectors from origin to the cloud
    for i in range(0, len(X), 3): 
        fig.add_trace(go.Scatter(x=[0, X[i, 0]], y=[0, X[i, 1]], mode='lines', line=dict(color='rgba(0, 123, 255, 0.1)', width=1), showlegend=False, hoverinfo='skip'))

    # 2. Scatter data
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', name='Data Points ($x_i$)', marker=dict(color='#007bff', size=6)))
    
    # 3. True Vector - Originates from Mean (Orange diagonal line)
    true_scale = 12
    fig.add_trace(go.Scatter(x=[mean_X[0] - pc1_true[0]*true_scale, mean_X[0] + pc1_true[0]*true_scale], 
                             y=[mean_X[1] - pc1_true[1]*true_scale, mean_X[1] + pc1_true[1]*true_scale], 
                             mode='lines', name='True PCA Axis (#ff9900)', line=dict(color='#ff9900', width=6)))

    # 4. Draw Mean Vector pointing in
    fig.add_trace(go.Scatter(x=[0, mean_X[0]], y=[0, mean_X[1]], mode='lines+markers', name='Mean Vector ($m$)', line=dict(color='yellow', width=5), marker=dict(size=12, symbol='arrow-bar-down', angleref='previous')))
    
    # 5. Draw Uncentered PCA Vector (Emphasize it overlapping the Mean)
    scale = np.linalg.norm(mean_X) * 1.15
    fig.add_trace(go.Scatter(x=[0, pc1_nc[0] * scale], y=[0, pc1_nc[1] * scale], mode='lines', name='PCA Uncentered (Pseudo-PC1)', line=dict(color='red', width=3, dash='dash')))

    fig.update_layout(
        title=dict(text="Resultant Force Turns Pseudo-PCA Vector Into A Straight Arrow!", font=dict(color="white", size=16)),
        xaxis=dict(range=[-2, 22], constrain='domain', gridcolor="#333", tickfont_color="white", title="Net Cash Flow (X)"),
        yaxis=dict(range=[-2, 22], scaleanchor="x", scaleratio=1, gridcolor="#333", tickfont_color="white", title="Net Worth Change (Y)"),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(font_color="white", bgcolor='rgba(0,0,0,0.5)', x=0.02, y=0.98),
        height=550
    )
    
    fig.add_hline(y=0, line_color="#555")
    fig.add_vline(x=0, line_color="#555")
    
    return fig

# ==========================================
# ADVANCED FUNCTION: Step-by-Step PCA Error (Section 3.2)
# ==========================================
def create_uncentered_step_by_step_plot():
    np.random.seed(42)
    # 1. Create a flat elongated cloud with negative slope, located in the 1st quadrant
    # Skewed diagonal Covariance matrix (with negative covariance)
    # (Var X large, Var Y medium, Cov deep negative)
    cov_skew = np.array([[8.0, -6.5], [-6.5, 6.0]])
    center_shift = np.array([12, 12])
    
    X_raw = np.random.multivariate_normal(center_shift, cov_skew, 150)
    mean_orig = np.mean(X_raw, axis=0)
    
    # Calculate True PCA Vector (correct)
    cov_true = np.cov(X_raw.T)
    evals_true, evecs_true = np.linalg.eig(cov_true)
    idx_t = evals_true.argsort()[::-1]
    evecs_true = evecs_true[:, idx_t]
    pc1_true = evecs_true[:, 0]
    
    # 2. Build error scatter matrix (Skip Zero-Center)
    pseudo_cov = (X_raw.T @ X_raw) / (len(X_raw) - 1)
    eigenvalues, eigenvectors = np.linalg.eig(pseudo_cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate a giant Ellipse originating from the origin
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    ellipse = (transform @ circle) * np.sqrt(chi2.ppf(0.95, 2))
    
    # Pseudo-PC1 vector pointing straight to the center
    scale = 1.05
    pc1 = eigenvectors[:, 0] * np.sqrt(eigenvalues[0]) * scale
    pc1_axis = eigenvectors[:, 0] * 35 
    
    v1 = eigenvectors[:, 0]
    proj_pts = (X_raw @ v1).reshape(-1, 1) * v1
    
    lt_x, lt_y = [], []
    for i in range(len(X_raw)):
        lt_x.extend([X_raw[i, 0], proj_pts[i, 0], None])
        lt_y.extend([X_raw[i, 1], proj_pts[i, 1], None])

    fig = go.Figure()
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.2)")

    null_val = [None]
    null_t = ["", ""]
    
    # Calculate the True PC1 line diagonally
    true_pc1_line_x = [mean_orig[0] - pc1_true[0]*12, mean_orig[0] + pc1_true[0]*12]
    true_pc1_line_y = [mean_orig[1] - pc1_true[1]*12, mean_orig[1] + pc1_true[1]*12]

    fig.add_trace(go.Scatter(x=X_raw[:,0], y=X_raw[:,1], mode='markers', name='Data', marker=dict(color='#007bff', size=7, opacity=0.7))) 
    fig.add_trace(go.Scatter(x=[mean_orig[0]], y=[mean_orig[1]], mode='markers', name='Mean', marker=dict(color='yellow', size=14, symbol='star'))) 
    
    # Add faint True PC1 trace underneath
    fig.add_trace(go.Scatter(x=true_pc1_line_x, y=true_pc1_line_y, mode='lines', name='True PC1 (Actual dispersion direction)', line=dict(color='#ff9900', width=5, dash='solid'), opacity=0.8)) 

    fig.add_trace(go.Scatter(x=null_val, y=null_val, mode='lines', name='Virtual Ellipse', line=dict(color='#E1BEE7', dash='dash', width=2))) 
    fig.add_trace(go.Scatter(x=null_val, y=null_val, mode='lines+markers+text', name='Error: Pseudo-PC1', line=dict(color='red', width=5), marker=dict(size=10, symbol='arrow-bar-up', angleref='previous'), textfont=dict(color="#ff3333", size=16, family="Arial Black"), textposition="top left")) 
    fig.add_trace(go.Scatter(x=null_val, y=null_val, mode='lines', name='Error projection line', line=dict(color='#aaaaaa', dash='dash', width=1), showlegend=False)) 
    
    fig.add_trace(go.Scatter(x=[12], y=[28], mode="text", name='Instructions', text=[""], textfont=dict(size=18, color="#ff4d4d"), showlegend=False))

    steps_data = [
        ("Original", X_raw, null_val, null_val, null_val, null_val, null_val, null_val, null_val, null_val, null_t, "<b>STEP 1: SKEWED DATA (UNCENTERED)</b> - True dispersion vector (Orange PC1) goes diagonally."),
        ("Covariance", X_raw, true_pc1_line_x, true_pc1_line_y, ellipse[0,:], ellipse[1,:], null_val, null_val, null_val, null_val, null_t, "<b>STEP 2: CALCULATE $X^T X$</b> - Valid virtual ellipse grows from original (0,0) instead of center (The enemy appears)."),
        ("PCA Vectors", X_raw, true_pc1_line_x, true_pc1_line_y, ellipse[0,:], ellipse[1,:], [0, pc1[0]], [0, pc1[1]], null_val, null_val, ["", "<b>PSEUDO PC1</b>"], "<b>STEP 3: FIND PC1</b> - Algorithm points straight at right angle to center, completely ignoring true axis!"),
        ("Projection", proj_pts, null_val, null_val, null_val, null_val, [-pc1_axis[0], pc1_axis[0]], [-pc1_axis[1], pc1_axis[1]], lt_x, lt_y, null_t, "<b>STEP 4: ERROR PROJECTION</b> - Entire data is flattened indistinguishably! COMPLETE LOSS OF INFO!")
    ]

    frames = []
    slider_steps = []
    for i, (name, x_d, xtrue, ytrue, xe, ye, xpc1, ypc1, xdash, ydash, tpc1, txt) in enumerate(steps_data):
        frame = go.Frame(
            name=f"step_{i}",
            data=[
                go.Scatter(x=x_d[:,0], y=x_d[:,1]), # Data
                go.Scatter(x=[mean_orig[0]], y=[mean_orig[1]]), # Mean
                go.Scatter(x=xtrue, y=ytrue), # True PC1 (Orange)
                go.Scatter(x=xe, y=ye), # Ellipse
                go.Scatter(x=xpc1, y=ypc1, text=tpc1), # Red vector
                go.Scatter(x=xdash, y=ydash), # Dashed projection line
                go.Scatter(x=[12], y=[28], text=[txt]) # Note
            ]
        )
        frames.append(frame)
        slider_steps.append(dict(method="animate", args=[[f"step_{i}"], dict(mode="immediate", frame=dict(duration=1200, redraw=False), transition=dict(duration=1200, easing="cubic-in-out"))], label=f"Step {i+1}"))

    fig.frames = frames

    fig.update_layout(
        xaxis=dict(range=[-5, 30], constrain='domain', gridcolor="#333", tickfont_color="white"),
        yaxis=dict(range=[-5, 30], scaleanchor="x", scaleratio=1, gridcolor="#333", tickfont_color="white"),
        paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
        margin=dict(l=20, r=20, t=20, b=20), height=650, legend=dict(font_color="white", bgcolor='rgba(0,0,0,0.5)', x=0.01, y=0.95),
        sliders=[dict(active=0, font=dict(color="white"), bgcolor="#444", bordercolor="#666",
            currentvalue={"prefix": "Error Progress: ", "font": dict(color="#ff4d4d", size=16)}, pad={"t": 30}, steps=slider_steps)]
    )
    return fig