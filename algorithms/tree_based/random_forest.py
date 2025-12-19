"""
Random Forest Demo - Customer Churn Prediction
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_confusion_matrix, plot_feature_importance
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class RandomForestDemo:
    def __init__(self):
        self.explanation = get_explanation('random_forest')
        self.model = None
        
    def render(self):
        st.markdown(f"# 🌲 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/dsa/random-forest-classifier-using-scikit-learn/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'rf_results' in st.session_state:
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
                feature_names = list(X.columns)
            else:
                return
        else:
            n_samples = st.slider("Samples", 200, 1000, 500)
            df = DatasetGenerator.generate_churn_data(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X, y = prepare_features_target(df, 'churned')
            feature_names = list(X.columns)
        
        st.markdown("### Model Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("N Estimators", 10, 200, 100)
        with col2:
            max_depth = st.slider("Max Depth", 2, 20, 10)
        with col3:
            max_features = st.selectbox("Max Features", ['sqrt', 'log2', None])
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training Random Forest..."):
                results = self._train_model(X, y, n_estimators, max_depth, max_features, feature_names)
                st.session_state['rf_results'] = results
                st.success("✅ Done!")
    
    def _render_predict(self):
        """Render prediction interface for customer churn"""
        st.markdown("## 🔮 Customer Churn Prediction")
        
        if 'rf_results' not in st.session_state or 'rf_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        
        model = st.session_state['rf_model']
        feature_names = st.session_state.get('rf_feature_names', [])
        
        if not feature_names:
            st.warning("⚠️ Feature names not found. Please retrain the model.")
            return
        
        st.info(f"📊 Enter values for {len(feature_names)} features to predict churn risk:")
        
        # Feature hints with typical values and help text
        feature_hints = {
            'tenure': {'value': 24.0, 'help': '📅 Months as customer (0-72)', 'min': 0.0, 'max': 72.0},
            'monthly_charges': {'value': 50.0, 'help': '💵 Monthly bill ($20-$100)', 'min': 20.0, 'max': 100.0},
            'total_charges': {'value': 1200.0, 'help': '💰 Total spent ($0-$7000)', 'min': 0.0, 'max': 7000.0},
            'support_tickets': {'value': 2.0, 'help': '🎫 Support tickets raised (0-10)', 'min': 0.0, 'max': 20.0},
            'has_streaming': {'value': 1.0, 'help': '📺 Has streaming (0=No, 1=Yes)', 'min': 0.0, 'max': 1.0},
            'has_phone': {'value': 1.0, 'help': '📱 Has phone service (0=No, 1=Yes)', 'min': 0.0, 'max': 1.0},
            'contract_type_One year': {'value': 0.0, 'help': '📋 One-year contract (0/1)', 'min': 0.0, 'max': 1.0},
            'contract_type_Two year': {'value': 0.0, 'help': '📋 Two-year contract (0/1)', 'min': 0.0, 'max': 1.0},
            'payment_method_Credit card': {'value': 1.0, 'help': '💳 Credit card payment (0/1)', 'min': 0.0, 'max': 1.0},
            'payment_method_Electronic check': {'value': 0.0, 'help': '🏦 Electronic check (0/1)', 'min': 0.0, 'max': 1.0},
        }
        
        # Create dynamic input fields based on training features
        input_values = []
        cols = st.columns(2)
        
        for i, feature in enumerate(feature_names):
            with cols[i % 2]:
                hint = feature_hints.get(feature, {'value': 0.0, 'help': f'Enter value for {feature}', 'min': None, 'max': None})
                value = st.number_input(
                    f"{feature}",
                    value=hint['value'],
                    min_value=hint.get('min'),
                    max_value=hint.get('max'),
                    help=hint['help'],
                    key=f"rf_pred_{feature}"
                )
                input_values.append(value)
        
        if st.button("🎯 Predict Churn Risk", type="primary"):
            input_data = pd.DataFrame([input_values], columns=feature_names)
            
            proba = model.predict_proba(input_data)[0]
            prediction = model.predict(input_data)[0]
            
            st.markdown("---")
            st.markdown("### 📈 Churn Prediction")
            
            if prediction == 1:
                st.error(f"## ⚠️ HIGH CHURN RISK ({proba[1]*100:.1f}%)")
                st.warning("Consider retention strategies for this customer!")
            else:
                st.success(f"## ✅ LOW CHURN RISK ({proba[0]*100:.1f}% retention likelihood)")
            
            st.markdown(f"**Churn Risk Score:** {proba[1]*100:.1f}%")
            st.progress(proba[1])
    
    def _train_model(self, X, y, n_estimators, max_depth, max_features, feature_names):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       max_features=max_features, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Store in session state for predictions
        st.session_state['rf_model'] = model
        st.session_state['rf_feature_names'] = feature_names
        
        return {
            'y_test': y_test, 'y_pred': y_pred,
            'feature_importance': dict(zip(feature_names, model.feature_importances_)),
            'oob_score': model.oob_score_ if hasattr(model, 'oob_score_') else None
        }
    
    def _render_results(self):
        r = st.session_state['rf_results']
        
        st.metric("Accuracy", f"{accuracy_score(r['y_test'], r['y_pred']):.4f}")
        
        # Feature importance
        st.markdown("## 📊 Feature Importance")
        fig = plot_feature_importance(
            np.array(list(r['feature_importance'].keys())),
            np.array(list(r['feature_importance'].values())),
            "Feature Importance"
        )
        st.plotly_chart(fig, width='stretch')
        
        cm = confusion_matrix(r['y_test'], r['y_pred'])
        fig = plot_confusion_matrix(cm, ['Retained', 'Churned'], "Confusion Matrix")
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Implementation Code")
        
        st.markdown("### 📦 Required Dependencies")
        st.code("pip install numpy pandas scikit-learn matplotlib", language="bash")
        
        st.markdown("### 🐍 Full Python Code")
        
        code = '''# ============================================
# RANDOM FOREST - Customer Churn Prediction
# ============================================
# Run: pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. GENERATE SAMPLE DATA (or load your own)
# --------------------------------------------
np.random.seed(42)
n_samples = 1000

# Create synthetic customer churn data
tenure = np.random.randint(1, 72, n_samples)
monthly_charges = np.random.uniform(20, 100, n_samples)
total_charges = tenure * monthly_charges * np.random.uniform(0.8, 1.2, n_samples)
contract_type = np.random.randint(0, 3, n_samples)
payment_method = np.random.randint(0, 4, n_samples)
tech_support = np.random.randint(0, 2, n_samples)
online_security = np.random.randint(0, 2, n_samples)

# Churn logic
churn_prob = (0.3 - 0.005 * tenure + 0.003 * monthly_charges 
              - 0.1 * contract_type - 0.05 * tech_support)
churned = (churn_prob + np.random.normal(0, 0.2, n_samples)) > 0.2

df = pd.DataFrame({
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'contract_type': contract_type,
    'payment_method': payment_method,
    'tech_support': tech_support,
    'online_security': online_security,
    'churned': churned.astype(int)
})

print("Dataset Shape:", df.shape)
print(f"Churn Rate: {df['churned'].mean():.2%}")

# --------------------------------------------
# 2. PREPARE DATA
# --------------------------------------------
X = df.drop('churned', axis=1)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------
# 3. TRAIN THE MODEL
# --------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --------------------------------------------
# 4. EVALUATE THE MODEL
# --------------------------------------------
y_pred = model.predict(X_test)

print("\\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# --------------------------------------------
# 5. FEATURE IMPORTANCE
# --------------------------------------------
print("\\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)
importances = model.feature_importances_
for feature, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
    print(f"{feature:20s}: {imp:.4f}")

# --------------------------------------------
# 6. VISUALIZATION
# --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Feature importance
importance_df = pd.DataFrame({
    'feature': X.columns, 
    'importance': importances
}).sort_values('importance')
axes[0].barh(importance_df['feature'], importance_df['importance'])
axes[0].set_xlabel('Importance')
axes[0].set_title('Feature Importance')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Retained', 'Churned'])
axes[1].set_yticklabels(['Retained', 'Churned'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i, j], ha='center', va='center', fontsize=20)

plt.tight_layout()
plt.savefig('random_forest_results.png')
plt.show()

print("\\n✅ Results saved to 'random_forest_results.png'")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Complete Code",
            data=code,
            file_name="random_forest_complete.py",
            mime="text/plain"
        )
