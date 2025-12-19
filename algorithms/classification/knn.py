"""
K-Nearest Neighbors Demo - Movie Recommendation
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_confusion_matrix, plot_decision_boundary
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class KNNDemo:
    def __init__(self):
        self.explanation = get_explanation('knn')
        
    def render(self):
        st.markdown(f"# 🎬 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html#knn) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'knn_results' in st.session_state:
                self._render_results()
            else:
                st.info("👆 Run the demo first!")
    
    def _render_theory(self):
        st.markdown("## Theory")
        st.markdown(self.explanation['theory'])
        
        st.markdown("### 📏 Distance Metrics")
        cols = st.columns(3)
        with cols[0]:
            st.info("**Euclidean**\nStraight-line distance\nMost common")
        with cols[1]:
            st.info("**Manhattan**\nCity-block distance\nGrid-like data")
        with cols[2]:
            st.info("**Minkowski**\nGeneralized form\nTunable parameter")
        
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
                target_col = st.selectbox("Target Column", df.columns)
                X, y = prepare_features_target(df, target_col)
            else:
                return
        else:
            from sklearn.datasets import make_classification
            n_samples = st.slider("Samples", 200, 1000, 500)
            X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=5, 
                                       n_classes=3, random_state=42)
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
            y = pd.Series(y)
        
        st.markdown("### Model Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_neighbors = st.slider("K (Neighbors)", 1, 20, 5)
        with col2:
            weights = st.selectbox("Weights", ['uniform', 'distance'])
        with col3:
            metric = st.selectbox("Distance Metric", ['euclidean', 'manhattan', 'minkowski'])
        
        if st.button("🚀 Train Model", type="primary"):
            results = self._train_model(X, y, n_neighbors, weights, metric)
            st.session_state['knn_results'] = results
            st.success("✅ Done!")
    
    def _train_model(self, X, y, n_neighbors, weights, metric):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        
        # Find optimal K
        k_range = range(1, 21)
        accuracies = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            accuracies.append(accuracy_score(y_test, knn.predict(X_test_scaled)))
        
        st.session_state['knn_model'] = model
        st.session_state['knn_scaler'] = scaler
        st.session_state['knn_feature_names'] = list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        st.session_state['knn_classes'] = list(np.unique(y))
        
        return {
            'y_test': y_test, 'y_pred': y_pred,
            'k_range': list(k_range), 'accuracies': accuracies,
            'n_neighbors': n_neighbors
        }
    
    def _render_results(self):
        r = st.session_state['knn_results']
        
        st.markdown("## 📊 Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(r['y_test'], r['y_pred']):.4f}")
        with col2:
            st.metric("K Used", r['n_neighbors'])
        
        # K optimization
        st.markdown("## 📈 Finding Optimal K")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r['k_range'], y=r['accuracies'],
            mode='lines+markers', name='Accuracy'
        ))
        fig.add_vline(x=r['n_neighbors'], line_dash="dash", line_color="red",
                      annotation_text=f"K={r['n_neighbors']}")
        fig.update_layout(
            xaxis_title="K", yaxis_title="Accuracy",
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Confusion matrix
        cm = confusion_matrix(r['y_test'], r['y_pred'])
        labels = [str(i) for i in np.unique(r['y_test'])]
        fig = plot_confusion_matrix(cm, labels, "Confusion Matrix")
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
K-NEAREST NEIGHBORS (KNN) - CLASSIFICATION
Complete code for Google Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# ============================================================
# STEP 1: GENERATE CLASSIFICATION DATA
# ============================================================
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, 
                          n_classes=3, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
y = pd.Series(y)

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")

# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: KNN requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\\nTraining: {len(X_train)}, Testing: {len(X_test)}")

# ============================================================
# STEP 3: FIND OPTIMAL K
# ============================================================
print("\\n🔍 Finding optimal K...")
k_range = range(1, 21)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    accuracies.append(acc)
    print(f"  K={k:2d}: Accuracy={acc:.4f}")

optimal_k = k_range[np.argmax(accuracies)]
print(f"\\n🏆 Optimal K: {optimal_k} (Accuracy: {max(accuracies):.4f})")

# ============================================================
# STEP 4: TRAIN FINAL MODEL
# ============================================================
model = KNeighborsClassifier(n_neighbors=optimal_k, weights='uniform', metric='euclidean')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ============================================================
# STEP 5: EVALUATE
# ============================================================
print("\\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# ============================================================
# STEP 6: VISUALIZATION
# ============================================================
plt.figure(figsize=(12, 5))

# Plot 1: K vs Accuracy
plt.subplot(1, 2, 1)
plt.plot(k_range, accuracies, 'o-', color='#667eea', linewidth=2, markersize=8)
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('Finding Optimal K')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Confusion Matrix
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.show()

print("\\n✅ KNN model trained!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="knn_classification.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface for user input"""
        st.markdown("## 🔮 Make Predictions")
        
        if 'knn_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        st.info("🎯 Enter feature values to classify a new data point using K-Nearest Neighbors:")
        
        model = st.session_state['knn_model']
        scaler = st.session_state['knn_scaler']
        feature_names = st.session_state['knn_feature_names']
        classes = st.session_state['knn_classes']
        
        st.markdown("### Enter Feature Values")
        st.caption("💡 Features are standardized - values around -2 to +2 are typical after scaling")
        
        # Create input fields dynamically based on features
        input_values = []
        cols = st.columns(min(3, len(feature_names)))
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                val = st.number_input(
                    f"{feature}",
                    value=0.0,
                    min_value=-5.0,
                    max_value=5.0,
                    help=f"📊 Enter value for {feature} (-5 to +5 typical range)",
                    key=f"knn_pred_{i}"
                )
                input_values.append(val)
        
        if st.button("🎯 Predict Class", type="primary"):
            # Create input DataFrame
            input_data = pd.DataFrame([input_values], columns=feature_names)
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            st.markdown(f"## Predicted Class: **{prediction}**")
            
            # Show probabilities
            st.markdown("### 📊 Class Probabilities")
            prob_df = pd.DataFrame({
                'Class': classes,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            st.dataframe(prob_df, use_container_width=True)
            
            # Confidence indicator
            max_prob = max(probabilities)
            if max_prob >= 0.8:
                st.success(f"✅ High confidence prediction ({max_prob*100:.1f}%)")
            elif max_prob >= 0.5:
                st.info(f"📊 Moderate confidence ({max_prob*100:.1f}%)")
            else:
                st.warning(f"⚠️ Low confidence ({max_prob*100:.1f}%) - consider more neighbors")

