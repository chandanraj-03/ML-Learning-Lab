"""
DBSCAN Demo - Network Anomaly Detection
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_cluster_results
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset


class DBSCANDemo:
    def __init__(self):
        self.explanation = get_explanation('dbscan')
        
    def render(self):
        st.markdown(f"# 🔍 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#dbscan) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'dbscan_results' in st.session_state:
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
            n_samples = st.slider("Samples", 500, 2000, 1000)
            df = DatasetGenerator.generate_network_traffic(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X = df.values
        
        st.markdown("### Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("Epsilon (ε)", 0.1, 5.0, 0.5)
        with col2:
            min_samples = st.slider("Min Samples", 2, 20, 5)
        
        if st.button("🚀 Run Clustering", type="primary"):
            results = self._run_clustering(X, eps, min_samples)
            st.session_state['dbscan_results'] = results
            st.success("✅ Done!")
    
    def _run_clustering(self, X, eps, min_samples):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        st.session_state['dbscan_model'] = model
        st.session_state['dbscan_scaler'] = scaler
        st.session_state['dbscan_core_samples'] = model.core_sample_indices_
        st.session_state['dbscan_components'] = model.components_ if hasattr(model, 'components_') and model.components_ is not None else X_scaled[model.core_sample_indices_]
        st.session_state['dbscan_n_features'] = X.shape[1]
        
        return {
            'X': X_scaled, 'labels': labels,
            'n_clusters': n_clusters, 'n_noise': n_noise
        }
    
    def _render_results(self):
        r = st.session_state['dbscan_results']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clusters Found", r['n_clusters'])
        with col2:
            st.metric("Noise Points (Anomalies)", r['n_noise'])
        
        fig = plot_cluster_results(r['X'], r['labels'], title="DBSCAN Clustering (Gray = Anomalies)")
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Implementation Code")
        
        st.markdown("### 📦 Required Dependencies")
        st.code("pip install numpy pandas scikit-learn matplotlib", language="bash")
        
        st.markdown("### 🐍 Full Python Code")
        
        code = '''# ============================================
# DBSCAN - Network Anomaly Detection
# ============================================
# Run: pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. GENERATE SAMPLE DATA (or load your own)
# --------------------------------------------
np.random.seed(42)
n_samples = 1000

# Create synthetic network traffic data
# Normal traffic clusters
normal_1 = np.random.randn(300, 2) * 0.5 + [2, 2]
normal_2 = np.random.randn(300, 2) * 0.5 + [8, 8]
normal_3 = np.random.randn(300, 2) * 0.5 + [5, 2]

# Anomalies (scattered)
anomalies = np.random.uniform(0, 10, (100, 2))

X = np.vstack([normal_1, normal_2, normal_3, anomalies])
df = pd.DataFrame(X, columns=['bytes_per_sec', 'packets_per_sec'])

print("Dataset Shape:", df.shape)
print(df.describe())

# --------------------------------------------
# 2. SCALE THE DATA
# --------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# --------------------------------------------
# 3. APPLY DBSCAN
# --------------------------------------------
eps = 0.5        # Maximum distance between points in a cluster
min_samples = 5  # Minimum points to form a dense region

model = DBSCAN(eps=eps, min_samples=min_samples)
labels = model.fit_predict(X_scaled)

# --------------------------------------------
# 4. ANALYZE RESULTS
# --------------------------------------------
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("\\n" + "="*50)
print("DBSCAN RESULTS")
print("="*50)
print(f"Parameters: eps={eps}, min_samples={min_samples}")
print(f"Clusters found: {n_clusters}")
print(f"Noise points (anomalies): {n_noise}")
print(f"Anomaly ratio: {n_noise / len(labels):.2%}")

print("\\nCluster sizes:")
for cluster in set(labels):
    count = list(labels).count(cluster)
    if cluster == -1:
        print(f"  Anomalies: {count}")
    else:
        print(f"  Cluster {cluster}: {count}")

# --------------------------------------------
# 5. VISUALIZATION
# --------------------------------------------
plt.figure(figsize=(10, 6))

# Plot clusters
unique_labels = set(labels)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        # Noise points (anomalies) in red
        color = 'red'
        marker = 'x'
        label_name = 'Anomalies'
    else:
        marker = 'o'
        label_name = f'Cluster {label}'
    
    mask = labels == label
    plt.scatter(df.iloc[mask, 0], df.iloc[mask, 1], 
                c=[color], marker=marker, s=50, label=label_name, alpha=0.7)

plt.xlabel('Bytes per Second')
plt.ylabel('Packets per Second')
plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
plt.legend()
plt.tight_layout()
plt.savefig('dbscan_results.png')
plt.show()

print("\\n✅ Results saved to 'dbscan_results.png'")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Complete Code",
            data=code,
            file_name="dbscan_complete.py",
            mime="text/plain"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Anomaly Detection")
        
        if 'dbscan_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.info("💡 DBSCAN identifies anomalies as points that don't belong to any cluster (label = -1)")
        st.markdown("Enter feature values to check if a new point would be considered an anomaly:")
        
        scaler = st.session_state['dbscan_scaler']
        n_features = st.session_state['dbscan_n_features']
        
        input_values = []
        cols = st.columns(min(3, n_features))
        for i in range(n_features):
            with cols[i % 3]:
                val = st.number_input(f"Feature {i+1}", value=0.0, key=f"dbscan_pred_{i}")
                input_values.append(val)
        
        if st.button("🎯 Check Point", type="primary"):
            st.markdown("---")
            st.info("Note: DBSCAN doesn't have a built-in predict method. To classify new points, you would need to check distance to core samples or retrain with the new data.")
            st.markdown("Consider using **GMM** or **Isolation Forest** for online anomaly detection.")
