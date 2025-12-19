"""
Decision Tree Demo - Loan Approval Prediction
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_confusion_matrix, plot_feature_importance
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class DecisionTreeDemo:
    def __init__(self):
        self.explanation = get_explanation('decision_tree')
        
    def render(self):
        st.markdown(f"# 🌳 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/tree.html) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/decision-tree-implementation-python/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'dt_results' in st.session_state:
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
            df = DatasetGenerator.generate_loan_data(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X = df.drop('approved', axis=1)
            y = df['approved']
            feature_names = list(X.columns)
        
        st.markdown("### Model Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            max_depth = st.slider("Max Depth", 1, 20, 5)
        with col2:
            criterion = st.selectbox("Criterion", ['gini', 'entropy'])
        with col3:
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
        
        if st.button("🚀 Train Model", type="primary"):
            results = self._train_model(X, y, max_depth, criterion, min_samples_split, feature_names)
            st.session_state['dt_results'] = results
            st.success("✅ Done!")
    
    def _train_model(self, X, y, max_depth, criterion, min_samples_split, feature_names):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, 
                                       min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.session_state['dt_model'] = model
        st.session_state['dt_feature_names'] = feature_names
        
        return {
            'y_test': y_test, 'y_pred': y_pred,
            'feature_importance': dict(zip(feature_names, model.feature_importances_)),
            'tree_depth': model.get_depth(), 'n_leaves': model.get_n_leaves()
        }
    
    def _render_results(self):
        r = st.session_state['dt_results']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(r['y_test'], r['y_pred']):.4f}")
        with col2:
            st.metric("Tree Depth", r['tree_depth'])
        with col3:
            st.metric("N Leaves", r['n_leaves'])
        
        # Feature importance
        fig = plot_feature_importance(
            np.array(list(r['feature_importance'].keys())),
            np.array(list(r['feature_importance'].values())),
            "Feature Importance"
        )
        st.plotly_chart(fig, width='stretch')
        
        cm = confusion_matrix(r['y_test'], r['y_pred'])
        fig = plot_confusion_matrix(cm, ['Rejected', 'Approved'], "Confusion Matrix")
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
DECISION TREE - LOAN APPROVAL PREDICTION
Complete code for Google Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# STEP 1: GENERATE LOAN DATA
# ============================================================
def generate_loan_data(n_samples=500, random_state=42):
    np.random.seed(random_state)
    
    income = np.random.uniform(20000, 150000, n_samples)
    credit_score = np.random.uniform(300, 850, n_samples)
    debt_ratio = np.random.uniform(0, 0.6, n_samples)
    employment_years = np.random.uniform(0, 30, n_samples)
    loan_amount = np.random.uniform(5000, 50000, n_samples)
    
    # Approval based on features
    score = (credit_score/850 * 0.4 + 
             (income/150000) * 0.3 + 
             (1 - debt_ratio) * 0.2 + 
             (employment_years/30) * 0.1)
    approved = (score + np.random.normal(0, 0.1, n_samples)) > 0.5
    
    return pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'debt_ratio': debt_ratio,
        'employment_years': employment_years,
        'loan_amount': loan_amount,
        'approved': approved.astype(int)
    })

df = generate_loan_data(500)
print("Dataset Preview:")
print(df.head())
print(f"\\nApproval rate: {df['approved'].mean():.2%}")

# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
X = df.drop('approved', axis=1)
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\\nTraining: {len(X_train)}, Testing: {len(X_test)}")

# ============================================================
# STEP 3: COMPARE DIFFERENT DEPTHS
# ============================================================
print("\\n📈 COMPARING TREE DEPTHS:")
print("-" * 50)

depths = [1, 2, 3, 5, 10, None]
results = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, criterion='gini', random_state=42)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    results.append({'depth': d, 'train_acc': train_acc, 'test_acc': test_acc, 'leaves': model.get_n_leaves()})
    print(f"  Depth={str(d):>4}: Train={train_acc:.4f}, Test={test_acc:.4f}, Leaves={model.get_n_leaves()}")

best = max(results, key=lambda x: x['test_acc'])
print(f"\\n🏆 Best Depth: {best['depth']} (Test Accuracy={best['test_acc']:.4f})")

# ============================================================
# STEP 4: TRAIN FINAL MODEL
# ============================================================
model = DecisionTreeClassifier(max_depth=best['depth'], criterion='gini', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ============================================================
# STEP 5: FEATURE IMPORTANCE
# ============================================================
print("\\n📊 FEATURE IMPORTANCE:")
print("-" * 40)
for name, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:20s}: {imp:.4f}")

# ============================================================
# STEP 6: VISUALIZATION
# ============================================================
plt.figure(figsize=(14, 5))

# Plot 1: Feature Importance
plt.subplot(1, 2, 1)
importance = model.feature_importances_
sorted_idx = np.argsort(importance)
plt.barh(X.columns[sorted_idx], importance[sorted_idx], color='#667eea')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.grid(True, alpha=0.3, axis='x')

# Plot 2: Confusion Matrix
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ['Rejected', 'Approved'])
plt.yticks([0, 1], ['Rejected', 'Approved'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=16)

plt.tight_layout()
plt.show()

print(f"\\n📊 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\\n✅ Decision Tree model trained!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="decision_tree_loan_approval.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Loan Approval Prediction")
        
        if 'dt_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        
        model = st.session_state['dt_model']
        feature_names = st.session_state['dt_feature_names']
        
        st.markdown("### Enter Applicant Details")
        
        # Feature hints with typical values for loan data
        feature_hints = {
            'income': {'value': 60000.0, 'help': '💵 Annual income ($20k-$200k)', 'min': 20000.0, 'max': 200000.0},
            'debt': {'value': 25000.0, 'help': '💳 Current debt ($0-$100k)', 'min': 0.0, 'max': 100000.0},
            'credit_score': {'value': 700.0, 'help': '📊 Credit score (300-850)', 'min': 300.0, 'max': 850.0},
            'employment_years': {'value': 5.0, 'help': '💼 Years employed (0-30)', 'min': 0.0, 'max': 30.0},
            'loan_amount': {'value': 50000.0, 'help': '🏦 Requested loan ($5k-$500k)', 'min': 5000.0, 'max': 500000.0},
            'debt_to_income': {'value': 0.3, 'help': '📈 Debt/income ratio (0-1)', 'min': 0.0, 'max': 2.0},
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
                    key=f"dt_pred_{i}"
                )
                input_values.append(val)
        
        if st.button("🎯 Predict", type="primary"):
            input_data = pd.DataFrame([input_values], columns=feature_names)
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.success(f"## ✅ LOAN APPROVED ({proba[1]*100:.1f}% confidence)")
            else:
                st.error(f"## ❌ LOAN REJECTED ({proba[0]*100:.1f}% confidence)")
            st.progress(proba[1])
