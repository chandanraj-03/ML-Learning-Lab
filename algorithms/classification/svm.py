"""
SVM Demo - Parkinson's Disease Prediction
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_confusion_matrix
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class SVMDemo:
    def __init__(self):
        self.explanation = get_explanation('svm')
        self.model = None
        self.scaler = StandardScaler()
        
    def render(self):
        st.markdown(f"# 🧠 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/svm.html#svm) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm//)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'svm_results' in st.session_state:
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
                target_col = st.selectbox("Target Column", df.columns)
                X, y = prepare_features_target(df, target_col)
            else:
                return
        else:
            n_samples = st.slider("Samples", 100, 500, 300)
            df = DatasetGenerator.generate_parkinsons_data(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X = df.drop('has_parkinsons', axis=1)
            y = df['has_parkinsons']
        
        st.markdown("### Model Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'])
        with col2:
            C = st.slider("C", 0.1, 10.0, 1.0)
        with col3:
            gamma = st.select_slider("Gamma", options=['scale', 'auto', 0.001, 0.01, 0.1, 1.0])
        
        if st.button("🚀 Train Model", type="primary"):
            results = self._train_model(X, y, kernel, C, gamma)
            st.session_state['svm_results'] = results
            st.success("✅ Done!")
    
    def _render_predict(self):
        """Render prediction interface for Parkinson's detection"""
        st.markdown("## 🔮 Parkinson's Disease Prediction")
        
        if 'svm_results' not in st.session_state or 'svm_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        # Check if we have feature names stored
        if 'svm_feature_names' not in st.session_state:
            st.warning("⚠️ Please retrain the model to enable predictions!")
            return
        
        feature_names = st.session_state['svm_feature_names']
        
        st.success("✅ Model is trained and ready for predictions!")
        st.info(f"🧠 Enter values for the {len(feature_names)} features below:")
        
        # Dynamically create inputs based on feature names
        input_values = []
        cols = st.columns(2)
        
        for i, feature in enumerate(feature_names):
            with cols[i % 2]:
                if 'fo' in feature.lower() or 'freq' in feature.lower():
                    val = st.number_input(f"{feature}", 88.0, 260.0, 150.0, key=f"svm_pred_{feature}")
                elif 'jitter' in feature.lower():
                    val = st.slider(f"{feature}", 0.0, 0.03, 0.005, 0.001, key=f"svm_pred_{feature}")
                elif 'shimmer' in feature.lower():
                    val = st.slider(f"{feature}", 0.0, 0.12, 0.03, 0.005, key=f"svm_pred_{feature}")
                elif 'nhr' in feature.lower():
                    val = st.slider(f"{feature}", 0.0, 0.3, 0.02, 0.005, key=f"svm_pred_{feature}")
                elif 'rpde' in feature.lower():
                    val = st.slider(f"{feature}", 0.3, 0.7, 0.5, 0.01, key=f"svm_pred_{feature}")
                else:
                    val = st.number_input(f"{feature}", value=0.0, key=f"svm_pred_{feature}")
                input_values.append(val)
        
        if st.button("🎯 Predict", type="primary"):
            model = st.session_state['svm_model']
            scaler = st.session_state['svm_scaler']
            
            input_data = np.array([input_values])
            input_scaled = scaler.transform(input_data)
            
            proba = model.predict_proba(input_scaled)[0]
            prediction = model.predict(input_scaled)[0]
            
            st.markdown("---")
            st.markdown("### 🩺 Diagnosis Prediction")
            
            if prediction == 1:
                st.error(f"## ⚠️ Parkinson's Indicators Detected ({proba[1]*100:.1f}% confidence)")
                st.warning("This is a prediction based on voice features. Please consult a medical professional.")
            else:
                st.success(f"## ✅ No Parkinson's Indicators ({proba[0]*100:.1f}% confidence)")
            
            st.markdown(f"**Risk Score:** {proba[1]*100:.1f}%")
            st.progress(proba[1])
    
    def _train_model(self, X, y, kernel, C, gamma):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Store in session state for predictions
        st.session_state['svm_model'] = model
        st.session_state['svm_scaler'] = scaler
        st.session_state['svm_feature_names'] = list(X.columns)  # Store feature names
        
        return {
            'y_test': y_test, 'y_pred': y_pred,
            'n_support': sum(model.n_support_), 'kernel': kernel
        }
    
    def _render_results(self):
        r = st.session_state['svm_results']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(r['y_test'], r['y_pred']):.4f}")
        with col2:
            st.metric("Precision", f"{precision_score(r['y_test'], r['y_pred']):.4f}")
        with col3:
            st.metric("Recall", f"{recall_score(r['y_test'], r['y_pred']):.4f}")
        with col4:
            st.metric("Support Vectors", r['n_support'])
        
        cm = confusion_matrix(r['y_test'], r['y_pred'])
        fig = plot_confusion_matrix(cm, ['Healthy', 'Parkinson\'s'], "Confusion Matrix")
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Implementation Code")
        
        st.markdown("### 📦 Required Dependencies")
        st.code("pip install numpy pandas scikit-learn matplotlib", language="bash")
        
        st.markdown("### 🐍 Full Python Code")
        
        code = '''# ============================================
# SVM - Parkinson's Disease Prediction
# ============================================
# Run: pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. GENERATE SAMPLE DATA (or load your own)
# --------------------------------------------
np.random.seed(42)
n_samples = 300

# Create synthetic Parkinson's voice features
mdvp_fo = np.random.uniform(88, 260, n_samples)  # Avg vocal frequency
mdvp_jitter = np.random.uniform(0, 0.03, n_samples)  # Voice jitter
mdvp_shimmer = np.random.uniform(0, 0.12, n_samples)  # Voice shimmer
nhr = np.random.exponential(0.02, n_samples)  # Noise-to-harmonics ratio
rpde = np.random.uniform(0.3, 0.7, n_samples)  # Recurrence period density

# Parkinsons label based on features
score = 0.5 * mdvp_jitter*100 + 0.3 * mdvp_shimmer*10 + 0.2 * nhr*50
has_parkinsons = (score + np.random.normal(0, 0.3, n_samples)) > 0.5

df = pd.DataFrame({
    'mdvp_fo': mdvp_fo,
    'mdvp_jitter': mdvp_jitter,
    'mdvp_shimmer': mdvp_shimmer,
    'nhr': nhr,
    'rpde': rpde,
    'has_parkinsons': has_parkinsons.astype(int)
})

print("Dataset Shape:", df.shape)
print(f"Parkinson's ratio: {df['has_parkinsons'].mean():.2%}")

# --------------------------------------------
# 2. PREPARE DATA
# --------------------------------------------
X = df.drop('has_parkinsons', axis=1)
y = df['has_parkinsons']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# 3. TRAIN THE MODEL
# --------------------------------------------
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train_scaled, y_train)

# --------------------------------------------
# 4. EVALUATE THE MODEL
# --------------------------------------------
y_pred = model.predict(X_test_scaled)

print("\\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"Support Vectors: {sum(model.n_support_)}")

print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------
# 5. VISUALIZATION
# --------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ['Healthy', "Parkinson's"])
plt.yticks([0, 1], ['Healthy', "Parkinson's"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=20)
plt.tight_layout()
plt.savefig('svm_results.png')
plt.show()

print("\\n✅ Results saved to 'svm_results.png'")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Complete Code",
            data=code,
            file_name="svm_complete.py",
            mime="text/plain"
        )
