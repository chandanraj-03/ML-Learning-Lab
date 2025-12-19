"""
Ridge Regression Demo for ML Portfolio
Project: Student Exam Performance Prediction (Handling Multicollinearity)
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_regression_results, plot_coefficient_path
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class RidgeRegressionDemo:
    def __init__(self):
        self.explanation = get_explanation('ridge_regression')
        self.model = None
        self.scaler = StandardScaler()
        
    def render(self):
        st.markdown(f"# 🏔️ {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        st.markdown(self.explanation['description'])
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/what-is-ridge-regression/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'ridge_results' in st.session_state:
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
        
        # Multicollinearity explanation
        st.markdown("### 🔗 Understanding Multicollinearity")
        st.info("""
        **Multicollinearity** occurs when features are highly correlated with each other.
        This causes:
        - Unstable coefficient estimates
        - Large variance in predictions
        - Difficulty interpreting feature importance
        
        Ridge regression shrinks correlated coefficients together, stabilizing the model.
        """)
    
    def _render_demo(self):
        st.markdown("## Interactive Demo")
        
        data_source = st.radio(
            "Choose Data Source",
            ["🎲 Synthetic Data", "📁 Upload Your Own"],
            horizontal=True
        )
        
        if data_source == "📁 Upload Your Own":
            uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
            if uploaded_file is not None:
                try:
                    df = load_user_dataset(uploaded_file)
                    st.success(f"✅ Loaded {len(df)} rows")
                    target_col = st.selectbox("Select Target Column", df.columns)
                    feature_cols = st.multiselect(
                        "Select Features",
                        [c for c in df.columns if c != target_col]
                    )
                    if target_col and feature_cols:
                        X, y = prepare_features_target(df, target_col, feature_cols)
                    else:
                        return
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
            else:
                st.info("👆 Upload a dataset")
                return
        else:
            st.markdown("### Synthetic Student Performance Data")
            st.warning("⚠️ This dataset has correlated features (study_hours ↔ homework_hours). Watch how Ridge handles them!")
            
            n_samples = st.slider("Number of Samples", 100, 500, 400)
            df = DatasetGenerator.generate_student_data(n_samples)
            
            with st.expander("📊 Data Preview & Correlation"):
                st.dataframe(df.head())
                
                # Correlation heatmap
                corr = df.corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale='RdBu',
                    text=np.round(corr.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig.update_layout(
                    title="Feature Correlation Matrix",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, width='stretch')
            
            X = df.drop('exam_score', axis=1)
            y = df['exam_score']
        
        st.markdown("### Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        with col2:
            alpha = st.slider("Alpha (λ)", 0.01, 100.0, 1.0)
        with col3:
            auto_alpha = st.checkbox("Auto-tune Alpha", value=True)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training..."):
                results = self._train_model(X, y, test_size, alpha, auto_alpha)
                st.session_state['ridge_results'] = results
                st.success("✅ Done! Check Results tab.")
    
    def _train_model(self, X, y, test_size, alpha, auto_alpha):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if auto_alpha:
            alphas = np.logspace(-3, 3, 100)
            ridge_cv = RidgeCV(alphas=alphas, cv=5)
            ridge_cv.fit(X_train_scaled, y_train)
            best_alpha = ridge_cv.alpha_
            self.model = Ridge(alpha=best_alpha)
        else:
            best_alpha = alpha
            self.model = Ridge(alpha=alpha)
        
        self.model.fit(X_train_scaled, y_train)
        
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Regularization path
        alphas = np.logspace(-3, 3, 50)
        coefs = []
        for a in alphas:
            ridge = Ridge(alpha=a)
            ridge.fit(X_train_scaled, y_train)
            coefs.append(ridge.coef_)
        
        st.session_state['ridge_model'] = self.model
        st.session_state['ridge_scaler'] = self.scaler
        st.session_state['ridge_feature_names'] = list(X.columns)
        
        return {
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_train': y_train,
            'y_train_pred': y_train_pred,
            'best_alpha': best_alpha,
            'coefficients': dict(zip(X.columns, self.model.coef_)),
            'alphas': alphas,
            'coef_path': np.array(coefs),
            'feature_names': list(X.columns)
        }
    
    def _render_results(self):
        results = st.session_state['ridge_results']
        
        st.markdown("## 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        train_r2 = r2_score(results['y_train'], results['y_train_pred'])
        test_r2 = r2_score(results['y_test'], results['y_test_pred'])
        test_mse = mean_squared_error(results['y_test'], results['y_test_pred'])
        
        with col1:
            st.metric("Best Alpha", f"{results['best_alpha']:.4f}")
        with col2:
            st.metric("Train R²", f"{train_r2:.4f}")
        with col3:
            st.metric("Test R²", f"{test_r2:.4f}")
        with col4:
            st.metric("Test MSE", f"{test_mse:.2f}")
        
        st.markdown("## 📈 Coefficient Analysis")
        
        # Unlike Lasso, Ridge keeps all features
        coef_df = pd.DataFrame({
            'Feature': list(results['coefficients'].keys()),
            'Coefficient': list(results['coefficients'].values())
        }).sort_values('Coefficient', key=abs, ascending=True)
        
        fig = go.Figure(go.Bar(
            x=coef_df['Coefficient'],
            y=coef_df['Feature'],
            orientation='h',
            marker=dict(color=coef_df['Coefficient'], colorscale='RdYlBu')
        ))
        fig.update_layout(
            title="Feature Coefficients (All Features Retained, Shrunk)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("## 📉 Regularization Path")
        fig = plot_coefficient_path(
            results['alphas'],
            results['coef_path'],
            results['feature_names'],
            "Coefficient Shrinkage with Increasing λ"
        )
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("## 🎯 Predictions")
        fig = plot_regression_results(
            results['y_test'].values,
            results['y_test_pred'],
            "Actual vs Predicted Exam Scores"
        )
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
RIDGE REGRESSION - STUDENT EXAM PERFORMANCE PREDICTION
Complete code for Google Colab
Demonstrates handling multicollinearity with L2 regularization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 1: GENERATE STUDENT PERFORMANCE DATA
# ============================================================
def generate_student_data(n_samples=400, random_state=42):
    """Generate synthetic student data with correlated features (multicollinearity)"""
    np.random.seed(random_state)
    
    # Correlated features (multicollinearity): study_hours and homework_hours are related
    study_hours = np.random.uniform(1, 8, n_samples)
    homework_hours = study_hours * 0.8 + np.random.normal(0, 0.5, n_samples)  # Correlated!
    sleep_hours = np.random.uniform(5, 9, n_samples)
    attendance = np.random.uniform(0.6, 1.0, n_samples)
    previous_score = np.random.uniform(40, 100, n_samples)
    
    # Exam score based on features
    exam_score = (
        20 + 
        5 * study_hours + 
        3 * homework_hours + 
        2 * sleep_hours + 
        20 * attendance + 
        0.3 * previous_score +
        np.random.normal(0, 5, n_samples)
    )
    exam_score = np.clip(exam_score, 0, 100)
    
    return pd.DataFrame({
        'study_hours': study_hours,
        'homework_hours': homework_hours,  # Correlated with study_hours!
        'sleep_hours': sleep_hours,
        'attendance': attendance,
        'previous_score': previous_score,
        'exam_score': exam_score
    })

df = generate_student_data(n_samples=400)
print("Dataset Preview:")
print(df.head())
print(f"\\nDataset shape: {df.shape}")

# Check correlation (multicollinearity)
print("\\n📊 Correlation Matrix:")
print(df.corr().round(2))

# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
X = df.drop('exam_score', axis=1)
y = df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: Ridge requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ============================================================
# STEP 3: COMPARE DIFFERENT ALPHA VALUES
# ============================================================
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
results = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train_scaled))
    test_r2 = r2_score(y_test, model.predict(X_test_scaled))
    results.append({
        'alpha': alpha, 'train_r2': train_r2, 'test_r2': test_r2,
        'coefs': model.coef_.copy()
    })

print("\\n📈 ALPHA COMPARISON:")
print("-" * 50)
print(f"{'Alpha':>10} | {'Train R²':>10} | {'Test R²':>10}")
print("-" * 50)
for r in results:
    print(f"{r['alpha']:>10.2f} | {r['train_r2']:>10.4f} | {r['test_r2']:>10.4f}")

# ============================================================
# STEP 4: FIND OPTIMAL ALPHA USING CROSS-VALIDATION
# ============================================================
print("\\n🔍 Finding optimal alpha using cross-validation...")
alphas_cv = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas_cv, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Optimal alpha: {ridge_cv.alpha_:.4f}")
print(f"Test R² with optimal alpha: {r2_score(y_test, ridge_cv.predict(X_test_scaled)):.4f}")

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
plt.figure(figsize=(14, 10))

# Plot 1: Regularization Path
plt.subplot(2, 2, 1)
alphas_path = np.logspace(-3, 3, 50)
coefs_path = []
for a in alphas_path:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)
    coefs_path.append(ridge.coef_)
coefs_path = np.array(coefs_path)

for i, feature in enumerate(X.columns):
    plt.plot(alphas_path, coefs_path[:, i], label=feature)
plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('Coefficient Value')
plt.title('Regularization Path: Coefficient Shrinkage')
plt.legend(loc='upper right', fontsize=8)
plt.grid(True, alpha=0.3)

# Plot 2: Train vs Test R²
plt.subplot(2, 2, 2)
train_scores = [r['train_r2'] for r in results]
test_scores = [r['test_r2'] for r in results]
x_pos = range(len(alphas))
plt.plot(x_pos, train_scores, 'o-', color='#4ecdc4', label='Train R²')
plt.plot(x_pos, test_scores, 's-', color='#ff6b6b', label='Test R²')
plt.xticks(x_pos, [str(a) for a in alphas])
plt.xlabel('Alpha (λ)')
plt.ylabel('R² Score')
plt.title('Train vs Test Performance')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Actual vs Predicted
plt.subplot(2, 2, 3)
y_pred = ridge_cv.predict(X_test_scaled)
plt.scatter(y_test, y_pred, alpha=0.6, c='#667eea', s=40)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title(f'Ridge Predictions (α={ridge_cv.alpha_:.3f}, R²={r2_score(y_test, y_pred):.3f})')
plt.grid(True, alpha=0.3)

# Plot 4: Feature Coefficients
plt.subplot(2, 2, 4)
coefs = ridge_cv.coef_
colors = ['#4ecdc4' if c > 0 else '#ff6b6b' for c in coefs]
plt.barh(X.columns, coefs, color=colors)
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients (All Retained, Shrunk)')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# ============================================================
# STEP 6: PREDICTIONS
# ============================================================
print("\\n🔮 SAMPLE PREDICTIONS:")
print("-" * 50)
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    print(f"Student {i+1}: Actual={actual:.1f}, Predicted={predicted:.1f}")

print("\\n📊 KEY INSIGHT:")
print("-" * 50)
print("Ridge keeps ALL features but SHRINKS correlated coefficients.")
print("Notice how study_hours and homework_hours (correlated) have")
print("more balanced coefficients compared to unregularized regression.")

print("\\n✅ Ridge Regression model trained!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="ridge_regression_student_performance.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Make Predictions")
        
        if 'ridge_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        st.info("📚 Enter student study metrics to predict exam score (0-100):")
        
        model = st.session_state['ridge_model']
        scaler = st.session_state['ridge_scaler']
        feature_names = st.session_state['ridge_feature_names']
        
        # Feature hints with typical values for student data
        feature_hints = {
            'study_hours': {'default': 4.0, 'min': 1.0, 'max': 8.0, 'help': '📖 Daily study hours (1-8)'},
            'homework_hours': {'default': 3.0, 'min': 0.5, 'max': 6.0, 'help': '📝 Daily homework hours (0.5-6)'},
            'sleep_hours': {'default': 7.0, 'min': 5.0, 'max': 9.0, 'help': '😴 Daily sleep hours (5-9)'},
            'attendance': {'default': 0.85, 'min': 0.6, 'max': 1.0, 'help': '✅ Class attendance rate (0.6-1.0)'},
            'previous_score': {'default': 70.0, 'min': 40.0, 'max': 100.0, 'help': '📊 Previous exam score (40-100)'},
        }
        
        st.markdown("### Enter Feature Values")
        
        input_values = []
        cols = st.columns(min(3, len(feature_names)))
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                hint = feature_hints.get(feature, {'default': 0.0, 'min': None, 'max': None, 'help': f'Enter {feature}'})
                val = st.number_input(
                    f"{feature}",
                    value=hint['default'],
                    min_value=hint['min'],
                    max_value=hint['max'],
                    help=hint['help'],
                    key=f"ridge_pred_{i}"
                )
                input_values.append(val)
        
        if st.button("🎯 Predict Exam Score", type="primary"):
            input_data = pd.DataFrame([input_values], columns=feature_names)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            st.markdown("---")
            st.markdown(f"### 🎯 Predicted Exam Score: **{prediction:.1f}/100**")
            
            # Score interpretation
            if prediction >= 90:
                st.success("🌟 Excellent! Top performer territory!")
            elif prediction >= 75:
                st.success("👍 Good score! Above average performance.")
            elif prediction >= 60:
                st.info("📚 Passing score. Room for improvement.")
            else:
                st.warning("⚠️ Below average. Consider more study time.")

