"""
OPTICS Demo - IoT Sensor Outlier Detection
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_cluster_results
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset


class OPTICSDemo:
    def __init__(self):
        self.explanation = get_explanation('optics')
        
    def render(self):
        st.markdown(f"# 📡 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#optics)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'optics_results' in st.session_state:
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
            n_samples = st.slider("Samples", 500, 1500, 1000)
            df = DatasetGenerator.generate_iot_sensor_data(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X = df.values
        
        col1, col2 = st.columns(2)
        with col1:
            min_samples = st.slider("Min Samples", 2, 20, 5)
        with col2:
            xi = st.slider("Xi (Steepness)", 0.01, 0.5, 0.05)
        
        if st.button("🚀 Run OPTICS", type="primary"):
            with st.spinner("Running OPTICS..."):
                results = self._run_clustering(X, min_samples, xi)
                st.session_state['optics_results'] = results
                st.success("✅ Done!")
    
    def _run_clustering(self, X, min_samples, xi):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = OPTICS(min_samples=min_samples, xi=xi)
        labels = model.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        return {
            'X': X_scaled, 'labels': labels,
            'reachability': model.reachability_,
            'ordering': model.ordering_,
            'n_clusters': n_clusters, 'n_noise': n_noise
        }
    
    def _render_results(self):
        r = st.session_state['optics_results']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clusters Found", r['n_clusters'])
        with col2:
            st.metric("Outliers", r['n_noise'])
        
        # Reachability plot
        st.markdown("## 📊 Reachability Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=r['reachability'][r['ordering']],
            mode='lines', line=dict(color='#667eea')
        ))
        fig.update_layout(
            title="Reachability Plot (valleys = clusters)",
            xaxis_title="Point Order", yaxis_title="Reachability Distance",
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, width='stretch')
        
        fig = plot_cluster_results(r['X'], r['labels'], title="OPTICS Clusters")
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
OPTICS CLUSTERING - IOT SENSOR OUTLIER DETECTION
Complete code for Google Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 1: GENERATE IOT SENSOR DATA
# ============================================================
np.random.seed(42)

# Normal sensor readings (clusters)
normal_1 = np.random.randn(300, 2) * 0.5 + [2, 2]
normal_2 = np.random.randn(300, 2) * 0.5 + [8, 8]
normal_3 = np.random.randn(200, 2) * 0.3 + [5, 4]

# Outlier readings (sparse)
outliers = np.random.uniform(0, 10, (100, 2))

X = np.vstack([normal_1, normal_2, normal_3, outliers])
df = pd.DataFrame(X, columns=['temperature', 'humidity'])

print("Dataset shape:", df.shape)
print(f"Expected outliers: ~{100} points")

# ============================================================
# STEP 2: SCALE DATA
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ============================================================
# STEP 3: APPLY OPTICS
# ============================================================
print("\\n🔍 Running OPTICS clustering...")
optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)
labels = optics.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = list(labels).count(-1)

print(f"\\n📊 RESULTS:")
print(f"Clusters found: {n_clusters}")
print(f"Outliers detected: {n_outliers}")
print(f"Outlier ratio: {n_outliers/len(labels):.2%}")

# ============================================================
# STEP 4: ANALYZE CLUSTERS
# ============================================================
print("\\n📈 CLUSTER SIZES:")
for c in sorted(set(labels)):
    count = list(labels).count(c)
    if c == -1:
        print(f"  Outliers: {count}")
    else:
        print(f"  Cluster {c}: {count}")

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
plt.figure(figsize=(14, 5))

# Plot 1: Reachability Plot
plt.subplot(1, 3, 1)
reachability = optics.reachability_
ordering = optics.ordering_
plt.plot(reachability[ordering], color='#667eea', linewidth=0.5)
plt.xlabel('Point Order')
plt.ylabel('Reachability Distance')
plt.title('Reachability Plot (Valleys = Clusters)')
plt.grid(True, alpha=0.3)

# Plot 2: Cluster Visualization
plt.subplot(1, 3, 2)
colors = ['gray' if l == -1 else plt.cm.viridis(l/max(1, max(labels))) for l in labels]
plt.scatter(df['temperature'], df['humidity'], c=colors, alpha=0.6, s=30)
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('OPTICS Clusters (Gray = Outliers)')
plt.grid(True, alpha=0.3)

# Plot 3: Outliers Highlighted
plt.subplot(1, 3, 3)
outlier_mask = labels == -1
plt.scatter(df.loc[~outlier_mask, 'temperature'], df.loc[~outlier_mask, 'humidity'], 
           c='blue', label='Normal', alpha=0.5, s=20)
plt.scatter(df.loc[outlier_mask, 'temperature'], df.loc[outlier_mask, 'humidity'], 
           c='red', label='Outliers', alpha=0.8, s=40, marker='x')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Outlier Detection')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n✅ OPTICS clustering complete!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="optics_iot_outlier_detection.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Cluster Assignment")
        st.info("💡 OPTICS doesn't support prediction on new data points. It requires recomputing the entire ordering for new data.")
        st.markdown("For prediction on new points, consider using **DBSCAN** with the same parameters or **GMM**.")
