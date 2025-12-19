"""
Logistic Regression Demo - Email Spam Detection
"""
import numpy as np
import pandas as pd
import streamlit as st
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_confusion_matrix, plot_roc_curve
from utils.explanations import get_explanation
from utils.features import TrainingTimer, render_training_time, render_export_button, render_dataset_visualization
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target



class LogisticRegressionDemo:
    def __init__(self):
        self.explanation = get_explanation('logistic_regression')
        self.model = None
        self.scaler = StandardScaler()
        
    def render(self):
        st.markdown(f"# 📧 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'logreg_results' in st.session_state:
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
                target_col = st.selectbox("Target Column (binary)", df.columns)
                X, y = prepare_features_target(df, target_col)
            else:
                return
        else:
            n_samples = st.slider("Samples", 500, 2000, 1000)
            df = DatasetGenerator.generate_spam_data(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X = df.drop('is_spam', axis=1)
            y = df['is_spam']
        
        # Dataset Visualization (new feature)
        with st.expander("📈 Dataset Visualization", expanded=False):
            render_dataset_visualization(df, 'is_spam' if 'is_spam' in df.columns else None)
        
        st.markdown("### Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            C = st.slider("C (Inverse Regularization)", 0.01, 10.0, 1.0)
        with col2:
            solver = st.selectbox("Solver", ['lbfgs', 'liblinear', 'newton-cg', 'sag'])
        
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5)
        
        col_train, col_reset = st.columns([2, 1])
        with col_train:
            train_btn = st.button("🚀 Train Model", type="primary")
        with col_reset:
            if st.button("🔄 Reset Model"):
                for key in ['logreg_results', 'logreg_model', 'logreg_scaler', 'logreg_training_time']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ Model reset!")
                st.rerun()
        
        if train_btn:
            with TrainingTimer() as timer:
                results = self._train_model(X, y, C, solver, threshold)
            st.session_state['logreg_results'] = results
            st.session_state['logreg_training_time'] = timer.get_duration_str()
            st.success("✅ Done!")
            render_training_time(timer.get_duration_str())

    
    def _render_predict(self):
        """Render prediction interface for user input"""
        st.markdown("## 🔮 Spam Detection")
        
        if 'logreg_results' not in st.session_state or 'logreg_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        # Check if we have feature names stored
        if 'logreg_feature_names' not in st.session_state:
            st.warning("⚠️ Please retrain the model to enable predictions!")
            return
        
        feature_names = st.session_state['logreg_feature_names']
        
        st.success("✅ Model is trained and ready for predictions!")
        st.info(f"📧 Enter values for the {len(feature_names)} features below:")
        
        # Dynamically create sliders based on feature names
        input_values = []
        cols = st.columns(2)
        
        for i, feature in enumerate(feature_names):
            with cols[i % 2]:
                # Set reasonable defaults based on feature name
                if 'freq' in feature.lower() or 'word' in feature.lower():
                    val = st.slider(f"{feature}", 0.0, 5.0, 0.5, key=f"pred_{feature}")
                elif 'count' in feature.lower() or 'exclamation' in feature.lower():
                    val = st.slider(f"{feature}", 0, 20, 2, key=f"pred_{feature}")
                elif 'length' in feature.lower() or 'capital' in feature.lower():
                    val = st.slider(f"{feature}", 1, 50, 5, key=f"pred_{feature}")
                else:
                    val = st.number_input(f"{feature}", value=0.0, key=f"pred_{feature}")
                input_values.append(val)
        
        if st.button("🎯 Check Email", type="primary"):
            model = st.session_state['logreg_model']
            scaler = st.session_state['logreg_scaler']
            
            input_data = np.array([input_values])
            input_scaled = scaler.transform(input_data)
            
            proba = model.predict_proba(input_scaled)[0]
            prediction = model.predict(input_scaled)[0]
            
            st.markdown("---")
            st.markdown("### 📧 Prediction Result")
            
            if prediction == 1:
                st.error(f"## 🚫 SPAM ({proba[1]*100:.1f}% confidence)")
            else:
                st.success(f"## ✅ NOT SPAM ({proba[0]*100:.1f}% confidence)")
            
            # Probability gauge
            st.markdown(f"**Spam Probability:** {proba[1]*100:.1f}%")
            st.progress(proba[1])
    
    def _train_model(self, X, y, C, solver, threshold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(C=C, solver=solver, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Store in session state for predictions
        st.session_state['logreg_model'] = model
        st.session_state['logreg_scaler'] = scaler
        st.session_state['logreg_feature_names'] = list(X.columns)  # Store feature names
        
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        return {
            'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba,
            'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr),
            'coefficients': dict(zip(X.columns, model.coef_[0])),
            'threshold': threshold
        }
    
    def _render_results(self):
        r = st.session_state['logreg_results']
        
        # Show training time if available
        if 'logreg_training_time' in st.session_state:
            render_training_time(st.session_state['logreg_training_time'])
        
        st.markdown("## 📊 Classification Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(r['y_test'], r['y_pred']):.4f}")
        with col2:
            st.metric("Precision", f"{precision_score(r['y_test'], r['y_pred']):.4f}")
        with col3:
            st.metric("Recall", f"{recall_score(r['y_test'], r['y_pred']):.4f}")
        with col4:
            st.metric("F1 Score", f"{f1_score(r['y_test'], r['y_pred']):.4f}")
        
        # Export Results Button
        export_data = {
            'accuracy': accuracy_score(r['y_test'], r['y_pred']),
            'precision': precision_score(r['y_test'], r['y_pred']),
            'recall': recall_score(r['y_test'], r['y_pred']),
            'f1_score': f1_score(r['y_test'], r['y_pred']),
            'auc': r['auc'],
            'threshold': r['threshold']
        }
        render_export_button(export_data, "Logistic_Regression")
        
        col1, col2 = st.columns(2)
        with col1:
            cm = confusion_matrix(r['y_test'], r['y_pred'])
            fig = plot_confusion_matrix(cm, ['Not Spam', 'Spam'], "Confusion Matrix")
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = plot_roc_curve(r['fpr'], r['tpr'], r['auc'], "ROC Curve")
            st.plotly_chart(fig, width='stretch')
        
        # Feature importance
        st.markdown("## 📈 Feature Importance")
        coef_df = pd.DataFrame({
            'Feature': list(r['coefficients'].keys()),
            'Coefficient': list(r['coefficients'].values())
        }).sort_values('Coefficient', key=abs, ascending=True)
        
        fig = go.Figure(go.Bar(
            x=coef_df['Coefficient'], y=coef_df['Feature'],
            orientation='h', marker=dict(color=coef_df['Coefficient'], colorscale='RdYlBu')
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Implementation Code")
        
        st.markdown("### 📦 Required Dependencies")
        st.code("pip install numpy pandas scikit-learn matplotlib", language="bash")
        
        st.markdown("### 🐍 Full Python Code")
        
        code = '''# ============================================
# LOGISTIC REGRESSION - Email Spam Detection
# ============================================
# Run: pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. GENERATE SAMPLE DATA (or load your own)
# --------------------------------------------
np.random.seed(42)
n_samples = 1000

# Create synthetic email features
word_freq_free = np.random.exponential(0.5, n_samples)
word_freq_money = np.random.exponential(0.3, n_samples)
word_freq_win = np.random.exponential(0.2, n_samples)
capital_run_length = np.random.exponential(5, n_samples)
exclamation_count = np.random.poisson(2, n_samples)

# Spam based on features
spam_score = (0.3 * word_freq_free + 0.4 * word_freq_money + 
              0.3 * word_freq_win + 0.01 * capital_run_length + 
              0.05 * exclamation_count)
is_spam = (spam_score + np.random.normal(0, 0.3, n_samples)) > 0.5

df = pd.DataFrame({
    'word_freq_free': word_freq_free,
    'word_freq_money': word_freq_money,
    'word_freq_win': word_freq_win,
    'capital_run_length': capital_run_length,
    'exclamation_count': exclamation_count,
    'is_spam': is_spam.astype(int)
})

print("Dataset Shape:", df.shape)
print(f"Spam ratio: {df['is_spam'].mean():.2%}")

# --------------------------------------------
# 2. PREPARE DATA
# --------------------------------------------
X = df.drop('is_spam', axis=1)
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# 3. TRAIN THE MODEL
# --------------------------------------------
model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)

# --------------------------------------------
# 4. MAKE PREDICTIONS
# --------------------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# --------------------------------------------
# 5. EVALUATE THE MODEL
# --------------------------------------------
print("\\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

print("\\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------
# 6. VISUALIZATION
# --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
axes[0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Not Spam', 'Spam'])
axes[1].set_yticklabels(['Not Spam', 'Spam'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i, j], ha='center', va='center', fontsize=20)

plt.tight_layout()
plt.savefig('logistic_regression_results.png')
plt.show()

print("\\n✅ Results saved to 'logistic_regression_results.png'")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Complete Code",
            data=code,
            file_name="logistic_regression_complete.py",
            mime="text/plain"
        )
