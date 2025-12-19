"""
Lasso Regression Demo for ML Portfolio
Project: Marketing Channel Analysis (Feature Selection)
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

import sys
sys.path.append('..')
from utils.visualization import plot_regression_results, plot_residuals, plot_coefficient_path
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class LassoRegressionDemo:
    def __init__(self):
        self.explanation = get_explanation('lasso_regression')
        self.model = None
        self.scaler = StandardScaler()
        
    def render(self):
        """Render the complete Lasso Regression demo page"""
        st.markdown(f"# 🎯 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        st.markdown(self.explanation['description'])
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#lasso) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/what-is-lasso-regression/)")
        
        with tab2:
            self._render_demo()
        
        with tab3:
            self._render_predict()
        
        with tab4:
            self._render_code()
        
        with tab5:
            if 'lasso_results' in st.session_state:
                self._render_results()
            else:
                st.info("👆 Run the demo first to see results!")
    
    def _render_theory(self):
        """Render theory and explanation"""
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
        
        st.markdown("### 🎯 Use Cases")
        cols = st.columns(4)
        for i, use_case in enumerate(self.explanation['use_cases']):
            with cols[i % 4]:
                st.info(use_case)
    
    def _render_demo(self):
        """Render interactive demo"""
        st.markdown("## Interactive Demo")
        
        data_source = st.radio(
            "Choose Data Source",
            ["🎲 Synthetic Data", "📁 Upload Your Own"],
            horizontal=True
        )
        
        if data_source == "📁 Upload Your Own":
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel file",
                type=['csv', 'xlsx', 'xls']
            )
            
            if uploaded_file is not None:
                try:
                    df = load_user_dataset(uploaded_file)
                    st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                    
                    target_col = st.selectbox("Select Target Column (Y)", df.columns)
                    feature_cols = st.multiselect(
                        "Select Feature Columns (X)",
                        [c for c in df.columns if c != target_col],
                        default=[c for c in df.columns if c != target_col][:5]
                    )
                    
                    if target_col and feature_cols:
                        X, y = prepare_features_target(df, target_col, feature_cols)
                    else:
                        st.warning("Please select target and feature columns")
                        return
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    return
            else:
                st.info("👆 Upload a dataset to continue")
                return
        else:
            st.markdown("### Synthetic Marketing Data")
            st.info("💡 This dataset has 8 marketing channels, but only some impact sales. Lasso will identify them!")
            
            col1, col2 = st.columns(2)
            with col1:
                n_samples = st.slider("Number of Samples", 100, 500, 300)
            
            df = DatasetGenerator.generate_marketing_data(n_samples)
            
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10))
            
            X = df.drop('Sales', axis=1)
            y = df['Sales']
        
        # Model configuration
        st.markdown("### Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        with col2:
            alpha = st.slider("Alpha (Regularization)", 0.01, 100.0, 1.0, 0.01)
        with col3:
            auto_alpha = st.checkbox("Auto-tune Alpha (CV)", value=True)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training Lasso model..."):
                results = self._train_model(X, y, test_size, alpha, auto_alpha)
                st.session_state['lasso_results'] = results
                st.success("✅ Model trained! Check Results tab.")
    
    def _train_model(self, X, y, test_size, alpha, auto_alpha):
        """Train the Lasso model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if auto_alpha:
            # Cross-validation to find best alpha
            lasso_cv = LassoCV(cv=5, random_state=42)
            lasso_cv.fit(X_train_scaled, y_train)
            best_alpha = lasso_cv.alpha_
            self.model = Lasso(alpha=best_alpha)
        else:
            best_alpha = alpha
            self.model = Lasso(alpha=alpha)
        
        self.model.fit(X_train_scaled, y_train)
        
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Regularization path
        alphas = np.logspace(-3, 2, 50)
        coefs = []
        for a in alphas:
            lasso = Lasso(alpha=a, max_iter=10000)
            lasso.fit(X_train_scaled, y_train)
            coefs.append(lasso.coef_)
        
        st.session_state['lasso_model'] = self.model
        st.session_state['lasso_scaler'] = self.scaler
        st.session_state['lasso_feature_names'] = list(X.columns)
        
        return {
            'model': self.model,
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
        """Render results"""
        results = st.session_state['lasso_results']
        
        st.markdown("## 📊 Model Performance")
        
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
        
        # Feature Selection Results
        st.markdown("## 🎯 Feature Selection Results")
        
        coef_df = pd.DataFrame({
            'Feature': list(results['coefficients'].keys()),
            'Coefficient': list(results['coefficients'].values())
        })
        coef_df['Selected'] = coef_df['Coefficient'].abs() > 0.001
        coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart
            colors = ['#667eea' if s else '#aaa' for s in coef_df['Selected']]
            fig = go.Figure(go.Bar(
                x=coef_df['Coefficient'],
                y=coef_df['Feature'],
                orientation='h',
                marker_color=colors
            ))
            fig.update_layout(
                title="Feature Coefficients (Selected vs Eliminated)",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("### Selected Features")
            selected = coef_df[coef_df['Selected']]
            for _, row in selected.iterrows():
                st.success(f"✅ {row['Feature']}")
            
            st.markdown("### Eliminated Features")
            eliminated = coef_df[~coef_df['Selected']]
            for _, row in eliminated.iterrows():
                st.error(f"❌ {row['Feature']}")
        
        # Regularization path
        st.markdown("## 📈 Regularization Path")
        fig = plot_coefficient_path(
            results['alphas'],
            results['coef_path'],
            results['feature_names'],
            "Coefficient Path (how coefficients shrink with increasing α)"
        )
        st.plotly_chart(fig, width='stretch')
        
        # Predictions
        st.markdown("## 📉 Prediction Results")
        fig = plot_regression_results(
            results['y_test'].values,
            results['y_test_pred'],
            "Actual vs Predicted Sales"
        )
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
LASSO REGRESSION - MARKETING CHANNEL ANALYSIS (FEATURE SELECTION)
Complete code for Google Colab
Demonstrates automatic feature selection with L1 regularization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 1: GENERATE MARKETING DATA
# ============================================================
def generate_marketing_data(n_samples=300, random_state=42):
    """Generate synthetic marketing data with some irrelevant features"""
    np.random.seed(random_state)
    
    # Important channels (these affect sales)
    tv_ads = np.random.uniform(0, 100, n_samples)
    online_ads = np.random.uniform(0, 80, n_samples)
    social_media = np.random.uniform(0, 50, n_samples)
    
    # Less important or irrelevant channels
    print_ads = np.random.uniform(0, 30, n_samples)
    radio_ads = np.random.uniform(0, 40, n_samples)
    billboard = np.random.uniform(0, 20, n_samples)
    email_marketing = np.random.uniform(0, 60, n_samples)
    influencer = np.random.uniform(0, 25, n_samples)
    
    # Sales based on ONLY some channels (Lasso should find these!)
    sales = (
        500 + 
        3.0 * tv_ads +           # Important
        2.5 * online_ads +       # Important
        1.5 * social_media +     # Important
        0.1 * print_ads +        # Weak - should be eliminated
        0.0 * radio_ads +        # Irrelevant - should be eliminated
        0.0 * billboard +        # Irrelevant - should be eliminated
        0.5 * email_marketing +  # Moderate
        0.0 * influencer +       # Irrelevant - should be eliminated
        np.random.normal(0, 50, n_samples)
    )
    
    return pd.DataFrame({
        'TV_Ads': tv_ads,
        'Online_Ads': online_ads,
        'Social_Media': social_media,
        'Print_Ads': print_ads,
        'Radio_Ads': radio_ads,
        'Billboard': billboard,
        'Email_Marketing': email_marketing,
        'Influencer': influencer,
        'Sales': sales
    })

df = generate_marketing_data(n_samples=300)
print("Dataset Preview:")
print(df.head())
print(f"\\nDataset shape: {df.shape}")
print(f"\\n💡 8 marketing channels, but only some actually impact Sales!")

# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
X = df.drop('Sales', axis=1)
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: Lasso requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ============================================================
# STEP 3: TRAIN LASSO WITH CROSS-VALIDATION
# ============================================================
print("\\n🔍 Finding optimal alpha using cross-validation...")
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")

# ============================================================
# STEP 4: FEATURE SELECTION RESULTS
# ============================================================
print("\\n🎯 FEATURE SELECTION RESULTS:")
print("-" * 50)

selected_features = []
eliminated_features = []

for feature, coef in zip(X.columns, lasso_cv.coef_):
    if abs(coef) > 0.001:
        selected_features.append((feature, coef))
        print(f"✅ {feature:20s}: {coef:>10.4f} (SELECTED)")
    else:
        eliminated_features.append(feature)
        print(f"❌ {feature:20s}: {coef:>10.4f} (ELIMINATED)")

print(f"\\n📊 Summary: {len(selected_features)} selected, {len(eliminated_features)} eliminated")

# ============================================================
# STEP 5: MODEL PERFORMANCE
# ============================================================
y_pred = lasso_cv.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\\n📈 MODEL PERFORMANCE:")
print("-" * 40)
print(f"Test R² Score: {test_r2:.4f}")
print(f"Test RMSE: ${test_rmse:.2f}")

# ============================================================
# STEP 6: VISUALIZATION
# ============================================================
plt.figure(figsize=(14, 10))

# Plot 1: Feature Coefficients (Selected vs Eliminated)
plt.subplot(2, 2, 1)
coefs = lasso_cv.coef_
colors = ['#4ecdc4' if abs(c) > 0.001 else '#cccccc' for c in coefs]
plt.barh(X.columns, coefs, color=colors)
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients (Green=Selected, Gray=Eliminated)')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')

# Plot 2: Regularization Path
plt.subplot(2, 2, 2)
alphas_path = np.logspace(-3, 1, 50)
coefs_path = []
for a in alphas_path:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    coefs_path.append(lasso.coef_)
coefs_path = np.array(coefs_path)

for i, feature in enumerate(X.columns):
    plt.plot(alphas_path, coefs_path[:, i], label=feature)
plt.xscale('log')
plt.xlabel('Alpha (λ)')
plt.ylabel('Coefficient Value')
plt.title('Regularization Path: Features Being Eliminated')
plt.legend(loc='upper right', fontsize=7)
plt.axvline(x=lasso_cv.alpha_, color='red', linestyle='--', label='Optimal α')
plt.grid(True, alpha=0.3)

# Plot 3: Actual vs Predicted
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred, alpha=0.6, c='#667eea', s=40)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title(f'Lasso Predictions (R²={test_r2:.3f})')
plt.grid(True, alpha=0.3)

# Plot 4: Feature Importance (Absolute)
plt.subplot(2, 2, 4)
importance = np.abs(lasso_cv.coef_)
sorted_idx = np.argsort(importance)
plt.barh(X.columns[sorted_idx], importance[sorted_idx], color='#667eea')
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance (Lasso)')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# ============================================================
# STEP 7: SAMPLE PREDICTIONS
# ============================================================
print("\\n🔮 SAMPLE PREDICTIONS:")
print("-" * 40)
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    print(f"Sample {i+1}: Actual=${actual:.0f}, Predicted=${predicted:.0f}")

print("\\n💡 KEY INSIGHT:")
print("-" * 50)
print("Lasso ELIMINATES irrelevant features (sets coefficients to 0).")
print("This is useful for:")
print("  • Identifying which marketing channels actually drive sales")
print("  • Building simpler, more interpretable models")
print("  • Reducing data collection costs (drop irrelevant features)")

print("\\n✅ Lasso Regression model trained!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="lasso_regression_marketing_analysis.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Make Predictions")
        
        if 'lasso_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        st.info("📊 Enter marketing spend ($) across channels to predict sales:")
        
        model = st.session_state['lasso_model']
        scaler = st.session_state['lasso_scaler']
        feature_names = st.session_state['lasso_feature_names']
        
        # Feature hints for marketing data
        feature_hints = {
            'TV_Ads': {'default': 50.0, 'min': 0.0, 'max': 100.0, 'help': '📺 TV advertising spend ($0-100k)'},
            'Online_Ads': {'default': 40.0, 'min': 0.0, 'max': 80.0, 'help': '💻 Online advertising spend ($0-80k)'},
            'Social_Media': {'default': 25.0, 'min': 0.0, 'max': 50.0, 'help': '📱 Social media spend ($0-50k)'},
            'Print_Ads': {'default': 15.0, 'min': 0.0, 'max': 30.0, 'help': '📰 Print advertising spend ($0-30k)'},
            'Radio_Ads': {'default': 20.0, 'min': 0.0, 'max': 40.0, 'help': '📻 Radio advertising spend ($0-40k)'},
            'Billboard': {'default': 10.0, 'min': 0.0, 'max': 20.0, 'help': '🪧 Billboard spend ($0-20k)'},
            'Email_Marketing': {'default': 30.0, 'min': 0.0, 'max': 60.0, 'help': '📧 Email marketing spend ($0-60k)'},
            'Influencer': {'default': 12.0, 'min': 0.0, 'max': 25.0, 'help': '🌟 Influencer marketing ($0-25k)'},
        }
        
        st.markdown("### Enter Marketing Spend")
        
        input_values = []
        cols = st.columns(min(3, len(feature_names)))
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                hint = feature_hints.get(feature, {'default': 0.0, 'min': 0.0, 'max': 100.0, 'help': f'Enter {feature}'})
                val = st.number_input(
                    f"{feature}",
                    value=hint['default'],
                    min_value=hint['min'],
                    max_value=hint['max'],
                    help=hint['help'],
                    key=f"lasso_pred_{i}"
                )
                input_values.append(val)
        
        if st.button("🎯 Predict Sales", type="primary"):
            input_data = pd.DataFrame([input_values], columns=feature_names)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            st.markdown("---")
            st.markdown(f"### 🎯 Predicted Sales: **${prediction:,.2f}**")
            
            # Show which channels are contributing (based on non-zero coefficients)
            st.markdown("### 📈 Channel Impact")
            coefs = model.coef_
            impact_data = []
            for feat, coef, val in zip(feature_names, coefs, input_values):
                if abs(coef) > 0.001:
                    impact_data.append({'Channel': feat, 'Coefficient': coef, 'Spend': val, 'Impact': coef * val})
            
            if impact_data:
                impact_df = pd.DataFrame(impact_data).sort_values('Impact', ascending=False)
                st.dataframe(impact_df, use_container_width=True)
            else:
                st.info("Model has eliminated most features - try adjusting alpha.")

