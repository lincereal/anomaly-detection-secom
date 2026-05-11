# ============================================
# ANOMALY DETECTION APP — SECOM Manufacturing
# Aldo Avila | github.com/lincereal
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA

# ── Configuración de la página ──────────────
st.set_page_config(
    page_title = "Anomaly Detector — SECOM",
    page_icon  = "🔧",
    layout     = "wide"
)

# ── Carga de modelos ─────────────────────────
@st.cache_resource
def load_models():
    with open('src/model/isolation_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('src/model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('src/model/pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open('src/model/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, pca, feature_cols

@st.cache_data
def load_data():
    X       = pd.read_csv('data/X_processed.csv')
    y       = pd.read_csv('data/y_labels.csv')['label']
    preds   = pd.read_csv('data/predictions.csv')
    return X, y, preds

model, scaler, pca, feature_cols = load_models()
X, y, preds = load_data()

# ── Header ───────────────────────────────────
st.title("🔧 Real-Time Anomaly Detection")
st.subheader("Semiconductor Manufacturing — SECOM Dataset")
st.markdown("---")

# ── Sidebar ──────────────────────────────────
st.sidebar.image(
    "https://img.shields.io/badge/Model-Isolation%20Forest-steelblue",
    use_container_width=True
)
st.sidebar.markdown("## ⚙️ Controls")

contamination = st.sidebar.slider(
    "Contamination threshold",
    min_value = 0.01,
    max_value = 0.30,
    value     = 0.15,
    step      = 0.01,
    help      = "Expected % of anomalies in the data"
)

show_real_labels = st.sidebar.checkbox(
    "Show real labels comparison",
    value = True
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 About")
st.sidebar.markdown("""
**Aldo Yamil Avila Carrillo**  
Quality Engineer → ML Engineer  
Sony Electronics | Master's in AI  
[GitHub](https://github.com/lincereal)
""")

# ── Re-predice con el threshold del slider ───
from sklearn.ensemble import IsolationForest

@st.cache_data
def predict_with_contamination(contamination_val):
    new_model = IsolationForest(
        n_estimators  = 200,
        contamination = contamination_val,
        random_state  = 42,
        n_jobs        = -1
    )
    new_model.fit(X)
    y_pred_new    = new_model.predict(X)
    scores_new    = new_model.score_samples(X)
    return y_pred_new, scores_new

y_pred_current, scores_current = predict_with_contamination(contamination)

# ── KPI Cards ────────────────────────────────
total     = len(X)
anomalies = (y_pred_current == -1).sum()
normal    = (y_pred_current ==  1).sum()
real_fails = (y == 1).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label = "🔍 Total Samples",
    value = f"{total:,}"
)
col2.metric(
    label = "🔴 Anomalies Detected",
    value = f"{anomalies}",
    delta = f"{anomalies/total*100:.1f}% of total"
)
col3.metric(
    label = "✅ Normal Products",
    value = f"{normal}",
    delta = f"{normal/total*100:.1f}% of total"
)
col4.metric(
    label = "⚠️ Real Failures in Dataset",
    value = f"{real_fails}",
    delta = f"{real_fails/total*100:.1f}% of total"
)

st.markdown("---")

# ── PCA Visualization ────────────────────────
st.markdown("## 📊 Anomaly Map — PCA Projection")
st.caption("297 sensors compressed to 2D for visualization")

X_2d = pca.transform(X)

if show_real_labels:
    col_left, col_right = st.columns(2)
else:
    col_left = st.columns(1)[0]

# Gráfica izquierda — predicciones del modelo
with col_left:
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig1.patch.set_facecolor('#0e1117')
    ax1.set_facecolor('#0e1117')

    colors_pred = ['#ff4b4b' if p == -1 else '#4b9eff'
                   for p in y_pred_current]
    ax1.scatter(X_2d[:, 0], X_2d[:, 1],
                c=colors_pred, alpha=0.5, s=15)
    ax1.set_title('Model Predictions', color='white', fontsize=12)
    ax1.set_xlabel('PC1', color='white')
    ax1.set_ylabel('PC2', color='white')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333')
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color='#ff4b4b', label='Anomaly'),
        Patch(color='#4b9eff', label='Normal')
    ], facecolor='#1e1e2e', labelcolor='white')
    ax1.grid(True, alpha=0.15, color='white')
    st.pyplot(fig1)
    plt.close()

# Gráfica derecha — etiquetas reales
if show_real_labels:
    with col_right:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')

        colors_real = ['#ff4b4b' if p == 1 else '#4b9eff'
                       for p in y]
        ax2.scatter(X_2d[:, 0], X_2d[:, 1],
                    c=colors_real, alpha=0.5, s=15)
        ax2.set_title('Real Labels', color='white', fontsize=12)
        ax2.set_xlabel('PC1', color='white')
        ax2.set_ylabel('PC2', color='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333')
        ax2.legend(handles=[
            Patch(color='#ff4b4b', label='Real Failure'),
            Patch(color='#4b9eff', label='OK')
        ], facecolor='#1e1e2e', labelcolor='white')
        ax2.grid(True, alpha=0.15, color='white')
        st.pyplot(fig2)
        plt.close()

st.markdown("---")

# ── Anomaly Score Distribution ───────────────
st.markdown("## 📈 Anomaly Score Distribution")

fig3, axes = plt.subplots(1, 2, figsize=(14, 4))
fig3.patch.set_facecolor('#0e1117')

for ax in axes:
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

# Histograma general
axes[0].hist(scores_current, bins=50,
             color='#4b9eff', edgecolor='none', alpha=0.85)
threshold = np.percentile(scores_current, contamination * 100)
axes[0].axvline(x=threshold, color='#ff4b4b',
                linewidth=2, linestyle='--',
                label=f'Threshold ({contamination*100:.0f}%)')
axes[0].set_title('Score Distribution', color='white')
axes[0].set_xlabel('Anomaly Score', color='white')
axes[0].set_ylabel('Count', color='white')
axes[0].legend(facecolor='#1e1e2e', labelcolor='white')
axes[0].grid(True, alpha=0.15, color='white')

# OK vs FAIL
axes[1].hist(scores_current[y == -1], bins=40,
             alpha=0.6, color='#4b9eff',
             label='OK Products', density=True)
axes[1].hist(scores_current[y ==  1], bins=40,
             alpha=0.6, color='#ff4b4b',
             label='FAIL Products', density=True)
axes[1].set_title('Scores: OK vs FAIL', color='white')
axes[1].set_xlabel('Anomaly Score', color='white')
axes[1].set_ylabel('Density', color='white')
axes[1].legend(facecolor='#1e1e2e', labelcolor='white')
axes[1].grid(True, alpha=0.15, color='white')

st.pyplot(fig3)
plt.close()

st.markdown("---")

# ── Tabla de productos anómalos ───────────────
st.markdown("## 🔴 Detected Anomalies — Detail View")

results_df = pd.DataFrame({
    'Sample ID'     : range(1, len(X) + 1),
    'Anomaly Score' : scores_current.round(4),
    'Prediction'    : ['🔴 ANOMALY' if p == -1
                       else '✅ NORMAL' for p in y_pred_current],
    'Real Label'    : ['❌ FAIL' if l == 1
                       else '✅ OK' for l in y]
})

# Filtros
col_f1, col_f2 = st.columns(2)
with col_f1:
    filter_pred = st.selectbox(
        "Filter by prediction:",
        ['All', '🔴 ANOMALY', '✅ NORMAL']
    )
with col_f2:
    filter_real = st.selectbox(
        "Filter by real label:",
        ['All', '❌ FAIL', '✅ OK']
    )

filtered = results_df.copy()
if filter_pred != 'All':
    filtered = filtered[filtered['Prediction'] == filter_pred]
if filter_real != 'All':
    filtered = filtered[filtered['Real Label'] == filter_real]

st.dataframe(
    filtered.sort_values('Anomaly Score').head(50),
    use_container_width = True,
    hide_index          = True
)

st.caption(f"Showing {len(filtered)} samples | "
           f"Sorted by anomaly score (most anomalous first)")

st.markdown("---")

# ── Footer ────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:gray; font-size:13px;'>
Built by <b>Aldo Yamil Avila Carrillo</b> | 
Quality Engineer → ML Engineer | 
Sony Electronics · Master's in AI<br>
<a href='https://github.com/lincereal/anomaly-detection-secom'>
GitHub Repository</a>
</div>
""", unsafe_allow_html=True)