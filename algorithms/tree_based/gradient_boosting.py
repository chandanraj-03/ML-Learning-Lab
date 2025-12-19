"""
Gradient Boosting Demo - Fraud Detection
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_confusion_matrix, plot_feature_importance, plot_roc_curve
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class GradientBoostingDemo:
    def __init__(self):
        self.explanation = get_explanation('gradient_boosting')
        
    def render(self):
        st.markdown(f"# 🚀 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/ml-gradient-boosting/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'gb_results' in st.session_state:
                self._render_results()
            else:
                st.info("👆 Run the demo first!")
    
    def _render_theory(self):
        st.markdown("## Theory")
        st.markdown(self.explanation['theory'])
        
        st.markdown("### 🔧 Popular Implementations")
        cols = st.columns(3)
        with cols[0]:
            st.success("**XGBoost**\nExtreme Gradient Boosting\nFast, regularized")
        with cols[1]:
            st.success("**LightGBM**\nLeaf-wise growth\nHistogram binning")
        with cols[2]:
            st.success("**CatBoost**\nOrdered boosting\nNative categorical")
        
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
                feature_names = list(X.columns)
            else:
                return
        else:
            n_samples = st.slider("Samples", 1000, 5000, 3000)
            df = DatasetGenerator.generate_fraud_data(n_samples)
            st.warning("⚠️ Fraud data is highly imbalanced (~2% fraud)")
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X = df.drop('is_fraud', axis=1)
            y = df['is_fraud']
            feature_names = list(X.columns)
        
        st.markdown("### Model Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("N Estimators", 50, 300, 100)
        with col2:
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
        with col3:
            max_depth = st.slider("Max Depth", 2, 10, 3)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training Gradient Boosting..."):
                results = self._train_model(X, y, n_estimators, learning_rate, max_depth, feature_names)
                st.session_state['gb_results'] = results
                st.success("✅ Done!")
    
    def _train_model(self, X, y, n_estimators, learning_rate, max_depth, feature_names):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                           max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        st.session_state['gb_model'] = model
        st.session_state['gb_feature_names'] = feature_names
        
        return {
            'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba,
            'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr),
            'feature_importance': dict(zip(feature_names, model.feature_importances_))
        }
    
    def _render_results(self):
        r = st.session_state['gb_results']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(r['y_test'], r['y_pred']):.4f}")
        with col2:
            st.metric("Precision", f"{precision_score(r['y_test'], r['y_pred'], zero_division=0):.4f}")
        with col3:
            st.metric("Recall", f"{recall_score(r['y_test'], r['y_pred']):.4f}")
        with col4:
            st.metric("AUC", f"{r['auc']:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_roc_curve(r['fpr'], r['tpr'], r['auc'], "ROC Curve")
            st.plotly_chart(fig, width='stretch')
        with col2:
            cm = confusion_matrix(r['y_test'], r['y_pred'])
            fig = plot_confusion_matrix(cm, ['Legit', 'Fraud'], "Confusion Matrix")
            st.plotly_chart(fig, width='stretch')
        
        fig = plot_feature_importance(
            np.array(list(r['feature_importance'].keys())),
            np.array(list(r['feature_importance'].values())),
            "Feature Importance"
        )
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
GRADIENT BOOSTING - FRAUD DETECTION
Complete code for Google Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# ============================================================
# STEP 1: GENERATE FRAUD DATA
# ============================================================
def generate_fraud_data(n_samples=3000, random_state=42):
    np.random.seed(random_state)
    
    # Generate features
    amount = np.random.exponential(200, n_samples)
    time_since_last = np.random.exponential(10, n_samples)
    distance = np.random.exponential(50, n_samples)
    transaction_type = np.random.randint(0, 5, n_samples)
    
    # Fraud logic (fraud is rare ~2%)
    fraud_score = (amount/1000 + distance/100 - time_since_last/20) / 3
    is_fraud = (fraud_score + np.random.normal(0, 0.3, n_samples)) > 0.8
    
    return pd.DataFrame({
        'amount': amount,
        'time_since_last': time_since_last,
        'distance': distance,
        'transaction_type': transaction_type,
        'is_fraud': is_fraud.astype(int)
    })

df = generate_fraud_data(3000)
print("Dataset shape:", df.shape)
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")

# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\\nTraining: {len(X_train)}, Testing: {len(X_test)}")

# ============================================================
# STEP 3: COMPARE LEARNING RATES
# ============================================================
print("\\n📈 COMPARING LEARNING RATES:")
print("-" * 50)

learning_rates = [0.01, 0.05, 0.1, 0.2]
results = []

for lr in learning_rates:
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({'lr': lr, 'accuracy': acc, 'f1': f1})
    print(f"  lr={lr:.2f}: Accuracy={acc:.4f}, F1={f1:.4f}")

best = max(results, key=lambda x: x['f1'])
print(f"\\n🏆 Best: lr={best['lr']} (F1={best['f1']:.4f})")

# ============================================================
# STEP 4: TRAIN FINAL MODEL
# ============================================================
model = GradientBoostingClassifier(n_estimators=100, learning_rate=best['lr'], max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ============================================================
# STEP 5: EVALUATE
# ============================================================
print("\\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# ============================================================
# STEP 6: VISUALIZATION
# ============================================================
plt.figure(figsize=(14, 5))

# Plot 1: ROC Curve
plt.subplot(1, 3, 1)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='#667eea', linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Feature Importance
plt.subplot(1, 3, 2)
importance = model.feature_importances_
sorted_idx = np.argsort(importance)
plt.barh(X.columns[sorted_idx], importance[sorted_idx], color='#4ecdc4')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.grid(True, alpha=0.3, axis='x')

# Plot 3: Confusion Matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ['Legit', 'Fraud'])
plt.yticks([0, 1], ['Legit', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.show()

print("\\n✅ Gradient Boosting model trained!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="gradient_boosting_fraud_detection.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Fraud Detection")
        
        if 'gb_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        
        model = st.session_state['gb_model']
        feature_names = st.session_state['gb_feature_names']
        
        st.markdown("### Enter Transaction Details")
        
        # Feature hints with typical values
        feature_hints = {
            'amount': {'value': 100.0, 'help': '💵 Transaction amount ($0-$5000)', 'min': 0.0, 'max': 10000.0},
            'time_of_day': {'value': 12.0, 'help': '🕐 Hour of day (0-24)', 'min': 0.0, 'max': 24.0},
            'distance_from_home': {'value': 15.0, 'help': '📍 Miles from home (0-200)', 'min': 0.0, 'max': 500.0},
            'hour_sin': {'value': 0.0, 'help': '📈 Time encoding (-1 to 1)', 'min': -1.0, 'max': 1.0},
            'hour_cos': {'value': 1.0, 'help': '📈 Time encoding (-1 to 1)', 'min': -1.0, 'max': 1.0},
        }
        
        input_values = []
        cols = st.columns(min(3, len(feature_names)))
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                hint = feature_hints.get(feature, {'value': 0.0, 'help': f'Enter value for {feature}', 'min': None, 'max': None})
                val = st.number_input(
                    f"{feature}",
                    value=hint['value'],
                    min_value=hint.get('min'),
                    max_value=hint.get('max'),
                    help=hint['help'],
                    key=f"gb_pred_{i}"
                )
                input_values.append(val)
        
        if st.button("🎯 Predict", type="primary"):
            input_data = pd.DataFrame([input_values], columns=feature_names)
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.error(f"## ⚠️ FRAUD DETECTED ({proba[1]*100:.1f}% probability)")
            else:
                st.success(f"## ✅ LEGITIMATE ({proba[0]*100:.1f}% confidence)")
            st.markdown(f"Fraud Probability: {proba[1]*100:.1f}%")
            st.progress(proba[1])
