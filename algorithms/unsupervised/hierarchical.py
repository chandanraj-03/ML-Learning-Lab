"""
Hierarchical Clustering Demo - Song Grouping
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from utils.visualization import plot_cluster_results
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset


class HierarchicalDemo:
    def __init__(self):
        self.explanation = get_explanation('hierarchical')
        
    def render(self):
        st.markdown(f"# 🎵 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'hier_results' in st.session_state:
                self._render_results()
            else:
                st.info("👆 Run the demo first!")
    
    def _render_theory(self):
        st.markdown("## Theory")
        st.markdown(self.explanation['theory'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ Pros")
            for pro in self.explanation['pros']:
                st.markdown(f"- {pro}")
        with col2:
            st.markdown("### ❌ Cons")
            for con in self.explanation['cons']:
                st.markdown(f"- {con}")
    
    def _render_demo(self):
        st.markdown("## Interactive Demo")
        
        data_source = st.radio("Data Source", ["🎲 Synthetic", "📁 Upload"], horizontal=True)
        
        if data_source == "📁 Upload":
            uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
            if uploaded_file:
                df = load_user_dataset(uploaded_file)
                feature_cols = st.multiselect("Select Features", df.select_dtypes(include=[np.number]).columns)
                if len(feature_cols) >= 2:
                    X = df[feature_cols].values
                else:
                    return
            else:
                return
        else:
            n_samples = st.slider("Samples", 100, 300, 200)
            df = DatasetGenerator.generate_song_features(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X = df.drop('genre', axis=1).values
        
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        with col2:
            linkage_type = st.selectbox("Linkage", ['ward', 'complete', 'average', 'single'])
        
        if st.button("🚀 Run Clustering", type="primary"):
            results = self._run_clustering(X, n_clusters, linkage_type)
            st.session_state['hier_results'] = results
            st.success("✅ Done!")
    
    def _run_clustering(self, X, n_clusters, linkage_type):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
        labels = model.fit_predict(X_scaled)
        
        # Linkage matrix for dendrogram
        Z = linkage(X_scaled[:50], method=linkage_type)  # Limit for visualization
        
        return {'X': X_scaled[:, :2], 'labels': labels, 'linkage': Z, 'n_clusters': n_clusters}
    
    def _render_results(self):
        r = st.session_state['hier_results']
        
        st.metric("Clusters", r['n_clusters'])
        
        # Dendrogram
        st.markdown("## 🌲 Dendrogram")
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f0f23')
        ax.set_facecolor('#0f0f23')
        dendrogram(r['linkage'], ax=ax, above_threshold_color='#667eea')
        ax.tick_params(colors='white')
        ax.set_title('Hierarchical Clustering Dendrogram', color='white')
        st.pyplot(fig)
        
        # Cluster scatter
        fig = plot_cluster_results(r['X'], r['labels'], title="Song Clusters")
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
HIERARCHICAL CLUSTERING - SONG GROUPING
Complete code for Google Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# ============================================================
# STEP 1: GENERATE SONG FEATURE DATA
# ============================================================
np.random.seed(42)

def generate_song_features(n_samples=200):
    # Generate song features for different genres
    pop = np.random.randn(n_samples//4, 4) * [0.1, 0.1, 0.1, 0.1] + [0.7, 0.5, 0.6, 0.4]
    rock = np.random.randn(n_samples//4, 4) * [0.1, 0.1, 0.1, 0.1] + [0.8, 0.7, 0.5, 0.6]
    jazz = np.random.randn(n_samples//4, 4) * [0.1, 0.1, 0.1, 0.1] + [0.3, 0.6, 0.8, 0.5]
    classical = np.random.randn(n_samples//4, 4) * [0.1, 0.1, 0.1, 0.1] + [0.2, 0.4, 0.9, 0.3]
    
    X = np.vstack([pop, rock, jazz, classical])
    X = np.clip(X, 0, 1)
    
    return pd.DataFrame(X, columns=['energy', 'tempo', 'complexity', 'loudness'])

df = generate_song_features(200)
print("Dataset shape:", df.shape)
print(df.describe().round(2))

# ============================================================
# STEP 2: SCALE DATA
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ============================================================
# STEP 3: CREATE LINKAGE MATRIX FOR DENDROGRAM
# ============================================================
print("\\n📊 Creating dendrogram...")
Z = linkage(X_scaled, method='ward')

# ============================================================
# STEP 4: AGGLOMERATIVE CLUSTERING
# ============================================================
n_clusters = 4
model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = model.fit_predict(X_scaled)

print(f"\\n📈 CLUSTER SIZES:")
for c in range(n_clusters):
    count = list(labels).count(c)
    print(f"  Cluster {c}: {count} songs")

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
plt.figure(figsize=(14, 10))

# Plot 1: Dendrogram
plt.subplot(2, 2, 1)
dendrogram(Z, truncate_mode='level', p=5, color_threshold=7)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')

# Plot 2: Cluster Scatter (Energy vs Tempo)
plt.subplot(2, 2, 2)
scatter = plt.scatter(df['energy'], df['tempo'], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Energy')
plt.ylabel('Tempo')
plt.title('Song Clusters (Energy vs Tempo)')
plt.colorbar(scatter, label='Cluster')

# Plot 3: Cluster Scatter (Complexity vs Loudness)
plt.subplot(2, 2, 3)
scatter = plt.scatter(df['complexity'], df['loudness'], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Complexity')
plt.ylabel('Loudness')
plt.title('Song Clusters (Complexity vs Loudness)')
plt.colorbar(scatter, label='Cluster')

# Plot 4: Cluster Profiles
plt.subplot(2, 2, 4)
df_with_labels = df.copy()
df_with_labels['cluster'] = labels
cluster_means = df_with_labels.groupby('cluster').mean()
cluster_means.T.plot(kind='bar', ax=plt.gca())
plt.xlabel('Feature')
plt.ylabel('Mean Value')
plt.title('Cluster Profiles')
plt.legend(title='Cluster')

plt.tight_layout()
plt.show()

print("\\n✅ Hierarchical clustering complete!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="hierarchical_song_clustering.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Cluster Assignment")
        st.info("💡 Hierarchical clustering doesn't natively support prediction on new data. To assign new points, you would need to either retrain or use a distance-based approach.")
        st.markdown("Consider using **GMM** or **K-Means** if you need prediction capabilities.")
