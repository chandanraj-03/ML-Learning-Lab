"""
K-Means Demo - Customer Segmentation
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_cluster_results, plot_elbow_curve
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset


class KMeansDemo:
    def __init__(self):
        self.explanation = get_explanation('kmeans')
        self.model = None
        self.scaler = StandardScaler()
        
    def render(self):
        st.markdown(f"# 🎯 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'kmeans_results' in st.session_state:
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
                    st.warning("Select at least 2 numeric features")
                    return
            else:
                return
        else:
            n_samples = st.slider("Samples", 200, 1000, 500)
            df = DatasetGenerator.generate_customer_segments(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X = df.values
        
        st.markdown("### Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters (K)", 2, 10, 5)
        with col2:
            init = st.selectbox("Initialization", ['k-means++', 'random'])
        
        find_optimal = st.checkbox("Find Optimal K (Elbow Method)")
        
        if st.button("🚀 Run Clustering", type="primary"):
            results = self._run_clustering(X, n_clusters, init, find_optimal)
            st.session_state['kmeans_results'] = results
            st.success("✅ Done!")
    
    def _render_predict(self):
        """Render prediction interface for customer segment assignment"""
        st.markdown("## 🔮 Customer Segment Prediction")
        
        if 'kmeans_results' not in st.session_state or 'kmeans_model' not in st.session_state:
            st.warning("⚠️ Please run clustering first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        st.info("👤 Enter customer data to find their segment:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 18, 80, 35, help="👤 Customer age (18-80 years)")
        with col2:
            income = st.number_input("Annual Income ($)", 10000, 200000, 50000, help="💰 Yearly income ($10k-$200k)")
        with col3:
            spending = st.number_input("Spending Score", 1, 2000, 500, help="🛒 Annual spending (1-2000 points)")
        
        if st.button("🎯 Find Segment", type="primary"):
            model = st.session_state['kmeans_model']
            scaler = st.session_state['kmeans_scaler']
            
            input_data = np.array([[age, income, spending]])
            input_scaled = scaler.transform(input_data)
            
            # Predict cluster
            cluster = model.predict(input_scaled)[0]
            
            # Get distance to all cluster centers
            distances = model.transform(input_scaled)[0]
            
            st.markdown("---")
            st.markdown("### 🎯 Assigned Segment")
            
            segment_names = ["Budget Conscious", "Young Professionals", "Premium Customers", "Steady Spenders", "High Value"]
            segment_name = segment_names[cluster % len(segment_names)]
            
            st.success(f"## Segment {cluster}: {segment_name}")
            
            # Show distance to all clusters
            st.markdown("### 📊 Distance to All Segments")
            dist_df = pd.DataFrame({
                'Segment': [f"Segment {i}" for i in range(len(distances))],
                'Distance': distances,
                'Match': ['✅ Best Match' if i == cluster else '' for i in range(len(distances))]
            }).sort_values('Distance')
            st.dataframe(dist_df, width='stretch')
    
    def _run_clustering(self, X, n_clusters, init, find_optimal):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = KMeans(n_clusters=n_clusters, init=init, n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
        
        # Store in session state for predictions
        st.session_state['kmeans_model'] = model
        st.session_state['kmeans_scaler'] = scaler
        
        results = {
            'X': X_scaled[:, :2] if X_scaled.shape[1] > 2 else X_scaled,
            'labels': labels,
            'centers': model.cluster_centers_[:, :2] if model.cluster_centers_.shape[1] > 2 else model.cluster_centers_,
            'inertia': model.inertia_,
            'silhouette': silhouette_score(X_scaled, labels)
        }
        
        if find_optimal:
            k_range = range(2, 11)
            inertias = []
            silhouettes = []
            for k in k_range:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(X_scaled)
                inertias.append(km.inertia_)
                silhouettes.append(silhouette_score(X_scaled, km.labels_))
            results['k_range'] = list(k_range)
            results['inertias'] = inertias
            results['silhouettes'] = silhouettes
        
        return results
    
    def _render_results(self):
        r = st.session_state['kmeans_results']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Inertia", f"{r['inertia']:.2f}")
        with col2:
            st.metric("Silhouette Score", f"{r['silhouette']:.4f}")
        
        # Cluster visualization
        fig = plot_cluster_results(r['X'], r['labels'], r['centers'], "Customer Segments")
        st.plotly_chart(fig, width='stretch')
        
        # Elbow curve
        if 'k_range' in r:
            st.markdown("## 📈 Finding Optimal K")
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_elbow_curve(r['k_range'], r['inertias'], "Elbow Method")
                st.plotly_chart(fig, width='stretch')
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=r['k_range'], y=r['silhouettes'], mode='lines+markers'))
                fig.update_layout(title="Silhouette Score", xaxis_title="K", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Implementation Code")
        
        st.markdown("### 📦 Required Dependencies")
        st.code("pip install numpy pandas scikit-learn matplotlib", language="bash")
        
        st.markdown("### 🐍 Full Python Code")
        
        code = '''# ============================================
# K-MEANS CLUSTERING - Customer Segmentation
# ============================================
# Run: pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. GENERATE SAMPLE DATA (or load your own)
# --------------------------------------------
np.random.seed(42)
n_samples = 500

# Create synthetic customer data
customers = np.vstack([
    np.random.randn(100, 3) * [5, 5000, 50] + [25, 10000, 100],   # Young, low income
    np.random.randn(100, 3) * [5, 10000, 100] + [35, 50000, 500], # Mid, med income
    np.random.randn(100, 3) * [5, 15000, 150] + [50, 100000, 1000], # Senior, high income
    np.random.randn(100, 3) * [5, 8000, 80] + [28, 30000, 300],   # Young professional
    np.random.randn(100, 3) * [5, 12000, 120] + [45, 80000, 800], # Established
])

df = pd.DataFrame(customers, columns=['age', 'income', 'spending_score'])
print("Dataset Shape:", df.shape)
print(df.describe())

# --------------------------------------------
# 2. SCALE THE DATA
# --------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# --------------------------------------------
# 3. FIND OPTIMAL K (Elbow Method)
# --------------------------------------------
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

print("\\n" + "="*50)
print("FINDING OPTIMAL K")
print("="*50)
for k, (inertia, sil) in zip(K_range, zip(inertias, silhouettes)):
    print(f"K={k}: Inertia={inertia:.2f}, Silhouette={sil:.4f}")

# --------------------------------------------
# 4. TRAIN FINAL MODEL
# --------------------------------------------
optimal_k = 5  # Choose based on elbow/silhouette
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

print("\\n" + "="*50)
print(f"FINAL MODEL (K={optimal_k})")
print("="*50)
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")

# Cluster centers (original scale)
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
print("\\nCluster Centers (original scale):")
print(pd.DataFrame(centers_original, columns=df.columns))

# Cluster sizes
print("\\nCluster Sizes:")
unique, counts = np.unique(labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} customers")

# --------------------------------------------
# 5. VISUALIZATION
# --------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Elbow curve
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
axes[0].legend()

# Silhouette scores
axes[1].plot(K_range, silhouettes, 'go-')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')

# Cluster visualization (2D projection)
axes[2].scatter(df['age'], df['income'], c=labels, cmap='viridis', alpha=0.6)
axes[2].scatter(centers_original[:, 0], centers_original[:, 1], 
                c='red', marker='X', s=200, edgecolors='black')
axes[2].set_xlabel('Age')
axes[2].set_ylabel('Income')
axes[2].set_title('Customer Segments')

plt.tight_layout()
plt.savefig('kmeans_results.png')
plt.show()

print("\\n✅ Results saved to 'kmeans_results.png'")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Complete Code",
            data=code,
            file_name="kmeans_complete.py",
            mime="text/plain"
        )
