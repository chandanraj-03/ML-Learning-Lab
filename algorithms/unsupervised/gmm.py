"""
Gaussian Mixture Models Demo - Image Segmentation
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_cluster_results
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset


class GMMDemo:
    def __init__(self):
        self.explanation = get_explanation('gmm')
        
    def render(self):
        st.markdown(f"# 🎨 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/mixture.html#mixture) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/gaussian-mixture-model/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'gmm_results' in st.session_state:
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
            df = DatasetGenerator.generate_image_colors(n_samples)
            with st.expander("📊 Data Preview (RGB Colors)"):
                st.dataframe(df.head())
            X = df.values
        
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider("Number of Components", 2, 10, 4)
        with col2:
            covariance_type = st.selectbox("Covariance Type", ['full', 'tied', 'diag', 'spherical'])
        
        if st.button("🚀 Run GMM", type="primary"):
            results = self._run_clustering(X, n_components, covariance_type)
            st.session_state['gmm_results'] = results
            st.success("✅ Done!")
    
    def _run_clustering(self, X, n_components, covariance_type):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
        labels = model.fit_predict(X_scaled)
        probs = model.predict_proba(X_scaled)
        
        st.session_state['gmm_model'] = model
        st.session_state['gmm_scaler'] = scaler
        st.session_state['gmm_n_features'] = X.shape[1]
        
        return {
            'X': X_scaled[:, :2], 'labels': labels, 'probs': probs,
            'bic': model.bic(X_scaled), 'aic': model.aic(X_scaled)
        }
    
    def _render_results(self):
        r = st.session_state['gmm_results']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("BIC", f"{r['bic']:.2f}")
        with col2:
            st.metric("AIC", f"{r['aic']:.2f}")
        
        fig = plot_cluster_results(r['X'], r['labels'], title="GMM Clusters")
        st.plotly_chart(fig, width='stretch')
        
        st.info("💡 GMM provides soft clustering - each point has a probability of belonging to each cluster")
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
GAUSSIAN MIXTURE MODELS (GMM) - IMAGE SEGMENTATION
Complete code for Google Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 1: GENERATE COLOR DATA (SIMULATING IMAGE PIXELS)
# ============================================================
np.random.seed(42)

# Create clusters of colors (RGB values)
colors = np.vstack([
    np.random.randn(200, 3) * 20 + [255, 100, 100],  # Reddish
    np.random.randn(200, 3) * 20 + [100, 255, 100],  # Greenish
    np.random.randn(200, 3) * 20 + [100, 100, 255],  # Bluish
    np.random.randn(200, 3) * 30 + [200, 200, 100],  # Yellowish
])
colors = np.clip(colors, 0, 255)

df = pd.DataFrame(colors, columns=['R', 'G', 'B'])
print("Dataset shape:", df.shape)
print(df.describe().round(1))

# ============================================================
# STEP 2: SCALE DATA
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ============================================================
# STEP 3: FIND OPTIMAL NUMBER OF COMPONENTS
# ============================================================
print("\\n🔍 Finding optimal number of components...")
n_components_range = range(2, 10)
bic_scores = []
aic_scores = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

optimal_n = n_components_range[np.argmin(bic_scores)]
print(f"\\n🏆 Optimal components (BIC): {optimal_n}")

# ============================================================
# STEP 4: TRAIN FINAL MODEL
# ============================================================
gmm = GaussianMixture(n_components=optimal_n, covariance_type='full', random_state=42)
labels = gmm.fit_predict(X_scaled)
probs = gmm.predict_proba(X_scaled)

print(f"\\n📊 RESULTS:")
print(f"BIC: {gmm.bic(X_scaled):.2f}")
print(f"AIC: {gmm.aic(X_scaled):.2f}")
print(f"Log-likelihood: {gmm.score(X_scaled):.4f}")

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
plt.figure(figsize=(14, 5))

# Plot 1: BIC/AIC Comparison
plt.subplot(1, 3, 1)
plt.plot(n_components_range, bic_scores, 'o-', label='BIC')
plt.plot(n_components_range, aic_scores, 's--', label='AIC')
plt.axvline(x=optimal_n, color='red', linestyle=':', label=f'Optimal={optimal_n}')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.title('Model Selection')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Cluster Assignments (2D projection)
plt.subplot(1, 3, 2)
scatter = plt.scatter(df['R'], df['G'], c=labels, cmap='viridis', alpha=0.6, s=30)
plt.xlabel('Red')
plt.ylabel('Green')
plt.title('GMM Clusters (R vs G)')
plt.colorbar(scatter, label='Cluster')

# Plot 3: Cluster Assignment Confidence
plt.subplot(1, 3, 3)
max_probs = np.max(probs, axis=1)
plt.hist(max_probs, bins=30, color='#667eea', edgecolor='white')
plt.xlabel('Max Assignment Probability')
plt.ylabel('Count')
plt.title('Cluster Assignment Confidence')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\\n✅ GMM clustering complete!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="gmm_image_segmentation.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Assign New Point to Cluster")
        
        if 'gmm_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained! Enter feature values to find cluster assignment.")
        
        model = st.session_state['gmm_model']
        scaler = st.session_state['gmm_scaler']
        n_features = st.session_state['gmm_n_features']
        
        st.markdown("### Enter Feature Values")
        
        input_values = []
        cols = st.columns(min(3, n_features))
        for i in range(n_features):
            with cols[i % 3]:
                val = st.number_input(f"Feature {i+1}", value=0.0, key=f"gmm_pred_{i}")
                input_values.append(val)
        
        if st.button("🎯 Assign Cluster", type="primary"):
            input_data = np.array([input_values])
            input_scaled = scaler.transform(input_data)
            cluster = model.predict(input_scaled)[0]
            probs = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.markdown(f"### 🎯 Assigned to Cluster: **{cluster}**")
            
            st.markdown("### Cluster Probabilities")
            prob_df = pd.DataFrame({'Cluster': range(len(probs)), 'Probability': probs})
            st.dataframe(prob_df.sort_values('Probability', ascending=False), use_container_width=True)
