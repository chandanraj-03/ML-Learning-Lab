"""
SVR Demo - Stock Price Forecasting
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_regression_results
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class SVRDemo:
    def __init__(self):
        self.explanation = get_explanation('svr')
        
    def render(self):
        st.markdown(f"# 📈 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/svm.html#svr) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'svr_results' in st.session_state:
                self._render_results()
            else:
                st.info("👆 Run the demo first!")
    
    def _render_theory(self):
        st.markdown("## Theory")
        st.markdown(self.explanation['theory'])
        
        st.markdown("### 🔧 Kernel Types")
        cols = st.columns(4)
        with cols[0]:
            st.info("**Linear**\nSimple, fast\nLinear patterns")
        with cols[1]:
            st.info("**RBF**\nMost common\nComplex patterns")
        with cols[2]:
            st.info("**Polynomial**\nCurved patterns\nDegree control")
        with cols[3]:
            st.info("**Sigmoid**\nNeural-like\nRarely used")
        
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
            df = DatasetGenerator.generate_stock_data(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X = df.drop(['close_price', 'day'], axis=1)
            y = df['close_price']
        
        st.markdown("### Model Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'])
        with col2:
            C = st.slider("C (Regularization)", 0.1, 100.0, 1.0)
        with col3:
            epsilon = st.slider("Epsilon (ε)", 0.01, 1.0, 0.1)
        
        gamma = 'scale'
        if kernel in ['rbf', 'poly', 'sigmoid']:
            gamma = st.select_slider("Gamma", options=['scale', 'auto', 0.001, 0.01, 0.1, 1.0])
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training SVR (may take a moment)..."):
                results = self._train_model(X, y, kernel, C, epsilon, gamma)
                st.session_state['svr_results'] = results
                st.success("✅ Done!")
    
    def _train_model(self, X, y, kernel, C, epsilon, gamma):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        model.fit(X_train_scaled, y_train)
        
        st.session_state['svr_model'] = model
        st.session_state['svr_scaler'] = scaler
        st.session_state['svr_feature_names'] = list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        return {
            'y_test': y_test, 'y_test_pred': model.predict(X_test_scaled),
            'y_train': y_train, 'y_train_pred': model.predict(X_train_scaled),
            'n_support': len(model.support_), 'kernel': kernel
        }
    
    def _render_results(self):
        r = st.session_state['svr_results']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test R²", f"{r2_score(r['y_test'], r['y_test_pred']):.4f}")
        with col2:
            st.metric("Test MSE", f"{mean_squared_error(r['y_test'], r['y_test_pred']):.2f}")
        with col3:
            st.metric("Support Vectors", r['n_support'])
        
        fig = plot_regression_results(r['y_test'].values, r['y_test_pred'], f"SVR ({r['kernel']}) Predictions")
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
SUPPORT VECTOR REGRESSION (SVR) - STOCK PRICE FORECASTING
Complete code for Google Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 1: GENERATE STOCK PRICE DATA
# ============================================================
def generate_stock_data(n_samples=300, random_state=42):
    """Generate synthetic stock price data with technical indicators"""
    np.random.seed(random_state)
    
    days = np.arange(n_samples)
    
    # Generate price with trend and seasonality
    trend = 100 + days * 0.1
    seasonality = 10 * np.sin(2 * np.pi * days / 30)
    noise = np.random.normal(0, 5, n_samples)
    close_price = trend + seasonality + noise
    
    # Technical indicators
    volume = np.random.uniform(1e6, 5e6, n_samples)
    open_price = close_price + np.random.normal(0, 2, n_samples)
    high_price = np.maximum(open_price, close_price) + np.abs(np.random.normal(0, 2, n_samples))
    low_price = np.minimum(open_price, close_price) - np.abs(np.random.normal(0, 2, n_samples))
    
    # Moving averages
    ma_5 = pd.Series(close_price).rolling(5).mean().fillna(close_price[0]).values
    ma_20 = pd.Series(close_price).rolling(20).mean().fillna(close_price[0]).values
    
    return pd.DataFrame({
        'day': days,
        'open_price': open_price,
        'high_price': high_price,
        'low_price': low_price,
        'volume': volume / 1e6,  # Scale to millions
        'ma_5': ma_5,
        'ma_20': ma_20,
        'close_price': close_price
    })

df = generate_stock_data(n_samples=300)
print("Dataset Preview:")
print(df.head())
print(f"\\nDataset shape: {df.shape}")

# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
X = df.drop(['close_price', 'day'], axis=1)
y = df['close_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: SVR requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ============================================================
# STEP 3: COMPARE DIFFERENT KERNELS
# ============================================================
def train_svr(X_train, y_train, X_test, y_test, kernel, C=1.0, epsilon=0.1):
    model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma='scale')
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    return {
        'model': model, 'kernel': kernel,
        'train_r2': train_r2, 'test_r2': test_r2,
        'n_support': len(model.support_)
    }

kernels = ['linear', 'rbf', 'poly']
results = [train_svr(X_train_scaled, y_train, X_test_scaled, y_test, k) for k in kernels]

print("\\n📈 KERNEL COMPARISON:")
print("-" * 60)
print(f"{'Kernel':>10} | {'Train R²':>10} | {'Test R²':>10} | {'Support Vectors':>15}")
print("-" * 60)
for r in results:
    print(f"{r['kernel']:>10} | {r['train_r2']:>10.4f} | {r['test_r2']:>10.4f} | {r['n_support']:>15}")

best = max(results, key=lambda x: x['test_r2'])
print(f"\\n🏆 Best Kernel: {best['kernel']} (Test R² = {best['test_r2']:.4f})")

# ============================================================
# STEP 4: HYPERPARAMETER TUNING FOR BEST KERNEL
# ============================================================
print(f"\\n🔧 Tuning C parameter for {best['kernel']} kernel...")
C_values = [0.1, 1.0, 10.0, 100.0]
tuning_results = []
for C in C_values:
    r = train_svr(X_train_scaled, y_train, X_test_scaled, y_test, best['kernel'], C=C)
    tuning_results.append(r)
    print(f"  C={C:>5.1f}: Test R²={r['test_r2']:.4f}")

best_tuned = max(tuning_results, key=lambda x: x['test_r2'])

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
plt.figure(figsize=(14, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
y_pred = best_tuned['model'].predict(X_test_scaled)
plt.scatter(y_test, y_pred, alpha=0.6, c='#667eea', s=40)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'SVR Predictions ({best_tuned["kernel"]}, R²={best_tuned["test_r2"]:.3f})')
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(1, 2, 2)
residuals = y_test.values - y_pred
plt.hist(residuals, bins=20, color='#4ecdc4', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# STEP 6: PREDICTION SUMMARY
# ============================================================
print("\\n📊 PREDICTION METRICS:")
print("-" * 40)
print(f"R² Score: {best_tuned['test_r2']:.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred):.2f}")
print(f"Support Vectors: {best_tuned['n_support']} ({best_tuned['n_support']/len(X_train)*100:.1f}% of training data)")

print("\\n🔮 SAMPLE PREDICTIONS:")
print("-" * 40)
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    print(f"Day {i+1}: Actual=${actual:.2f}, Predicted=${predicted:.2f}")

print("\\n✅ SVR model trained and ready!")
'''
        st.code(code, language='python')
        
        # Download button
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="svr_stock_price_forecasting.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Make Predictions")
        
        if 'svr_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        st.info("📈 Enter stock market indicators to predict closing price:")
        
        model = st.session_state['svr_model']
        scaler = st.session_state['svr_scaler']
        feature_names = st.session_state['svr_feature_names']
        
        # Feature hints for stock data
        feature_hints = {
            'open_price': {'default': 105.0, 'min': 50.0, 'max': 200.0, 'help': '💵 Opening price ($50-200)'},
            'high_price': {'default': 108.0, 'min': 50.0, 'max': 210.0, 'help': '📈 Day high price ($50-210)'},
            'low_price': {'default': 102.0, 'min': 45.0, 'max': 200.0, 'help': '📉 Day low price ($45-200)'},
            'volume': {'default': 2.5, 'min': 1.0, 'max': 5.0, 'help': '📊 Trading volume (millions)'},
            'ma_5': {'default': 104.0, 'min': 50.0, 'max': 200.0, 'help': '📏 5-day moving average'},
            'ma_20': {'default': 102.0, 'min': 50.0, 'max': 200.0, 'help': '📐 20-day moving average'},
        }
        
        st.markdown("### Enter Stock Indicators")
        
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
                    key=f"svr_pred_{i}"
                )
                input_values.append(val)
        
        if st.button("🎯 Predict Closing Price", type="primary"):
            input_data = pd.DataFrame([input_values], columns=feature_names)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            st.markdown("---")
            st.markdown(f"### 🎯 Predicted Closing Price: **${prediction:.2f}**")
            
            # Price movement indicator
            if 'open_price' in feature_names:
                open_idx = feature_names.index('open_price')
                open_price = input_values[open_idx]
                change = prediction - open_price
                pct_change = (change / open_price) * 100
                if change > 0:
                    st.success(f"📈 Up ${change:.2f} ({pct_change:+.2f}%) from open")
                else:
                    st.error(f"📉 Down ${abs(change):.2f} ({pct_change:.2f}%) from open")

