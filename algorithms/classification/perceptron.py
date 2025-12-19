"""
Perceptron Demo - Handwritten Digit Classification (Odd vs Even)
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_confusion_matrix
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class PerceptronDemo:
    def __init__(self):
        self.explanation = get_explanation('perceptron')
        
    def render(self):
        st.markdown(f"# ✏️ {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#perceptron) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/what-is-perceptron-the-simplest-artificial-neural-network/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'perceptron_results' in st.session_state:
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
        
        data_source = st.radio("Data Source", ["🎲 Digits Data", "📁 Upload"], horizontal=True)
        
        if data_source == "📁 Upload":
            uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
            if uploaded_file:
                df = load_user_dataset(uploaded_file)
                target_col = st.selectbox("Target Column (binary)", df.columns)
                X, y = prepare_features_target(df, target_col)
            else:
                return
        else:
            X, y = DatasetGenerator.generate_digit_data(500)
            st.info("🔢 Classifying handwritten digits as Odd or Even")
        
        col1, col2 = st.columns(2)
        with col1:
            eta0 = st.slider("Learning Rate (η)", 0.01, 2.0, 1.0)
        with col2:
            max_iter = st.slider("Max Iterations", 100, 2000, 1000)
        
        if st.button("🚀 Train Model", type="primary"):
            results = self._train_model(X, y, eta0, max_iter)
            st.session_state['perceptron_results'] = results
            st.success("✅ Done!")
    
    def _train_model(self, X, y, eta0, max_iter):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = Perceptron(eta0=eta0, max_iter=max_iter, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        st.session_state['perceptron_model'] = model
        st.session_state['perceptron_scaler'] = scaler
        st.session_state['perceptron_feature_names'] = list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        return {
            'y_test': y_test, 'y_pred': y_pred,
            'n_iter': model.n_iter_, 'n_features': X.shape[1]
        }
    
    def _render_results(self):
        r = st.session_state['perceptron_results']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(r['y_test'], r['y_pred']):.4f}")
        with col2:
            st.metric("Iterations", r['n_iter'])
        with col3:
            st.metric("Features", r['n_features'])
        
        cm = confusion_matrix(r['y_test'], r['y_pred'])
        fig = plot_confusion_matrix(cm, ['Even', 'Odd'], "Confusion Matrix")
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
PERCEPTRON - DIGIT CLASSIFICATION (ODD vs EVEN)
Complete code for Google Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

# ============================================================
# STEP 1: LOAD DIGIT DATA
# ============================================================
digits = load_digits()
X = digits.data
y = (digits.target % 2)  # 0=Even, 1=Odd

print(f"Dataset shape: {X.shape}")
print(f"Classes: Even (0), Odd (1)")
print(f"Class distribution: {np.bincount(y)}")

# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\\nTraining: {len(X_train)}, Testing: {len(X_test)}")

# ============================================================
# STEP 3: COMPARE LEARNING RATES
# ============================================================
print("\\n📈 COMPARING LEARNING RATES:")
print("-" * 40)

learning_rates = [0.01, 0.1, 0.5, 1.0, 2.0]
results = []

for eta in learning_rates:
    model = Perceptron(eta0=eta, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    results.append({'eta': eta, 'accuracy': acc, 'iterations': model.n_iter_})
    print(f"  η={eta:.2f}: Accuracy={acc:.4f}, Iterations={model.n_iter_}")

best = max(results, key=lambda x: x['accuracy'])
print(f"\\n🏆 Best: η={best['eta']} (Accuracy={best['accuracy']:.4f})")

# ============================================================
# STEP 4: TRAIN FINAL MODEL
# ============================================================
model = Perceptron(eta0=best['eta'], max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ============================================================
# STEP 5: EVALUATE
# ============================================================
print("\\n📊 PERFORMANCE:")
print("-" * 40)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Iterations: {model.n_iter_}")
print(f"\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Even', 'Odd']))

# ============================================================
# STEP 6: VISUALIZATION
# ============================================================
plt.figure(figsize=(12, 5))

# Plot 1: Learning Rate Comparison
plt.subplot(1, 2, 1)
etas = [r['eta'] for r in results]
accs = [r['accuracy'] for r in results]
plt.bar(range(len(etas)), accs, color='#667eea')
plt.xticks(range(len(etas)), [str(e) for e in etas])
plt.xlabel('Learning Rate (η)')
plt.ylabel('Accuracy')
plt.title('Perceptron: Learning Rate Comparison')
plt.grid(True, alpha=0.3, axis='y')

# Plot 2: Confusion Matrix
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ['Even', 'Odd'])
plt.yticks([0, 1], ['Even', 'Odd'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=16)

plt.tight_layout()
plt.show()

print("\\n✅ Perceptron model trained!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="perceptron_digit_classification.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Make Predictions")
        
        if 'perceptron_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        
        model = st.session_state['perceptron_model']
        scaler = st.session_state['perceptron_scaler']
        feature_names = st.session_state['perceptron_feature_names']
        
        st.markdown("### Enter Feature Values")
        
        input_values = []
        cols = st.columns(min(3, len(feature_names)))
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                val = st.number_input(f"{feature}", value=0.0, key=f"perceptron_pred_{i}")
                input_values.append(val)
        
        if st.button("🎯 Predict Class", type="primary"):
            input_data = pd.DataFrame([input_values], columns=feature_names)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            st.markdown("---")
            st.markdown(f"### 🎯 Predicted Class: **{'Even' if prediction == 0 else 'Odd'}**")
