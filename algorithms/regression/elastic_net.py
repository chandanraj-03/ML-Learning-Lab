"""
Elastic Net Demo - Car Price Prediction
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_regression_results
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class ElasticNetDemo:
    def __init__(self):
        self.explanation = get_explanation('elastic_net')
        
    def render(self):
        st.markdown(f"# 🔗 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/lasso-vs-ridge-vs-elastic-net-ml/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'elasticnet_results' in st.session_state:
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
        
        data_source = st.radio("Choose Data Source", ["🎲 Synthetic Data", "📁 Upload Your Own"], horizontal=True)
        
        if data_source == "📁 Upload Your Own":
            uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
            if uploaded_file:
                df = load_user_dataset(uploaded_file)
                target_col = st.selectbox("Target Column", df.columns)
                X, y = prepare_features_target(df, target_col)
            else:
                return
        else:
            st.markdown("### Synthetic Car Price Data")
            st.info("🚗 This dataset includes categorical (brand, fuel) and continuous (mileage, engine) features!")
            n_samples = st.slider("Samples", 200, 1000, 500)
            df = DatasetGenerator.generate_car_prices(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10))
                st.markdown("**Columns:**")
                st.write(list(df.columns))
            X, y = prepare_features_target(df, 'price')
        
        col1, col2 = st.columns(2)
        with col1:
            l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
        with col2:
            alpha = st.slider("Alpha", 0.01, 10.0, 1.0)
        
        if st.button("🚀 Train Model", type="primary"):
            results = self._train_model(X, y, alpha, l1_ratio)
            st.session_state['elasticnet_results'] = results
            st.success("✅ Done!")
    
    def _train_model(self, X, y, alpha, l1_ratio):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_train_scaled, y_train)
        
        st.session_state['elasticnet_model'] = model
        st.session_state['elasticnet_scaler'] = scaler
        st.session_state['elasticnet_feature_names'] = list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        return {
            'y_test': y_test, 'y_test_pred': model.predict(X_test_scaled),
            'y_train': y_train, 'y_train_pred': model.predict(X_train_scaled),
            'coefficients': dict(zip(X.columns, model.coef_)),
            'alpha': alpha, 'l1_ratio': l1_ratio
        }
    
    def _render_results(self):
        r = st.session_state['elasticnet_results']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test R²", f"{r2_score(r['y_test'], r['y_test_pred']):.4f}")
        with col2:
            st.metric("Alpha", f"{r['alpha']:.2f}")
        with col3:
            st.metric("L1 Ratio", f"{r['l1_ratio']:.2f}")
        
        fig = plot_regression_results(r['y_test'].values, r['y_test_pred'], "Predictions")
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
ELASTIC NET REGRESSION - CAR PRICE PREDICTION
Complete code for Google Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 1: GENERATE CAR PRICE DATA
# ============================================================
def generate_car_prices(n_samples=500, random_state=42):
    """Generate synthetic car price data with multiple features"""
    np.random.seed(random_state)
    
    # Generate features
    mileage = np.random.uniform(5000, 150000, n_samples)
    age = np.random.uniform(0, 15, n_samples)
    engine_size = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], n_samples)
    horsepower = 80 + engine_size * 40 + np.random.normal(0, 10, n_samples)
    fuel_efficiency = 35 - engine_size * 5 + np.random.normal(0, 2, n_samples)
    
    # Price based on features (with some noise)
    price = (
        25000 
        - mileage * 0.05 
        - age * 1500 
        + engine_size * 5000 
        + horsepower * 50
        + np.random.normal(0, 2000, n_samples)
    )
    price = np.clip(price, 3000, 80000)
    
    return pd.DataFrame({
        'mileage': mileage,
        'age': age,
        'engine_size': engine_size,
        'horsepower': horsepower,
        'fuel_efficiency': fuel_efficiency,
        'price': price
    })

df = generate_car_prices(n_samples=500)
print("Dataset Preview:")
print(df.head())
print(f"\\nDataset shape: {df.shape}")

# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for Elastic Net)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ============================================================
# STEP 3: TRAIN ELASTIC NET MODELS WITH DIFFERENT PARAMETERS
# ============================================================
def train_elastic_net(X_train, y_train, X_test, y_test, alpha, l1_ratio):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    n_nonzero = np.sum(model.coef_ != 0)
    return {
        'model': model, 'alpha': alpha, 'l1_ratio': l1_ratio,
        'train_r2': train_r2, 'test_r2': test_r2, 'n_features': n_nonzero
    }

# Compare different L1 ratios (0=Ridge, 1=Lasso, 0.5=Elastic Net)
l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
results = [train_elastic_net(X_train_scaled, y_train, X_test_scaled, y_test, 1.0, l1) for l1 in l1_ratios]

print("\\n📈 MODEL COMPARISON (varying L1 ratio, alpha=1.0):")
print("-" * 65)
print(f"{'L1 Ratio':>10} | {'Type':>12} | {'Train R²':>10} | {'Test R²':>10} | {'Features':>8}")
print("-" * 65)
for r in results:
    reg_type = "Ridge" if r['l1_ratio']==0 else ("Lasso" if r['l1_ratio']==1 else "Elastic")
    print(f"{r['l1_ratio']:>10.2f} | {reg_type:>12} | {r['train_r2']:>10.4f} | {r['test_r2']:>10.4f} | {r['n_features']:>8}")

best = max(results, key=lambda x: x['test_r2'])
print(f"\\n🏆 Best Model: L1 Ratio={best['l1_ratio']} (Test R² = {best['test_r2']:.4f})")

# ============================================================
# STEP 4: CROSS-VALIDATION TO FIND OPTIMAL ALPHA
# ============================================================
print("\\n🔍 Finding optimal alpha using cross-validation...")
elastic_cv = ElasticNetCV(l1_ratio=0.5, alphas=np.logspace(-4, 2, 50), cv=5, max_iter=10000)
elastic_cv.fit(X_train_scaled, y_train)
print(f"Optimal alpha: {elastic_cv.alpha_:.4f}")

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
plt.figure(figsize=(14, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
y_pred = best['model'].predict(X_test_scaled)
plt.scatter(y_test, y_pred, alpha=0.5, c='#667eea', s=40)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Elastic Net Predictions (L1={best["l1_ratio"]}, R²={best["test_r2"]:.3f})')
plt.grid(True, alpha=0.3)

# Plot 2: Feature Coefficients
plt.subplot(1, 2, 2)
coefs = best['model'].coef_
features = X.columns
colors = ['#4ecdc4' if c > 0 else '#ff6b6b' for c in coefs]
plt.barh(features, coefs, color=colors)
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Elastic Net Coefficients)')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# ============================================================
# STEP 6: MAKE PREDICTIONS
# ============================================================
print("\\n🔮 SAMPLE PREDICTIONS:")
print("-" * 50)
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    print(f"Car {i+1}: Actual=${actual:,.0f}, Predicted=${predicted:,.0f}, Error=${abs(actual-predicted):,.0f}")

# Predict for new car
new_car = pd.DataFrame({
    'mileage': [50000], 'age': [3], 'engine_size': [2.0],
    'horsepower': [180], 'fuel_efficiency': [28]
})
new_car_scaled = scaler.transform(new_car)
predicted_price = best['model'].predict(new_car_scaled)[0]
print(f"\\n🚗 New Car Prediction: ${predicted_price:,.0f}")

print("\\n✅ Model trained and ready!")
'''
        st.code(code, language='python')
        
        # Download button
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="elastic_net_car_price_prediction.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Make Predictions")
        
        if 'elasticnet_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        st.info("🚗 Enter car details to predict its price:")
        
        model = st.session_state['elasticnet_model']
        scaler = st.session_state['elasticnet_scaler']
        feature_names = st.session_state['elasticnet_feature_names']
        
        # Feature hints for car data
        feature_hints = {
            'mileage': {'default': 50000.0, 'min': 5000.0, 'max': 150000.0, 'help': '📏 Odometer reading (5k-150k miles)'},
            'age': {'default': 3.0, 'min': 0.0, 'max': 15.0, 'help': '📅 Car age in years (0-15)'},
            'engine_size': {'default': 2.0, 'min': 1.0, 'max': 4.0, 'help': '⚙️ Engine size in liters (1.0-4.0L)'},
            'horsepower': {'default': 180.0, 'min': 80.0, 'max': 350.0, 'help': '🐎 Horsepower (80-350 HP)'},
            'fuel_efficiency': {'default': 28.0, 'min': 15.0, 'max': 40.0, 'help': '⛽ MPG fuel efficiency (15-40)'},
        }
        
        st.markdown("### Enter Car Details")
        
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
                    key=f"elasticnet_pred_{i}"
                )
                input_values.append(val)
        
        if st.button("🎯 Predict Car Price", type="primary"):
            input_data = pd.DataFrame([input_values], columns=feature_names)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            st.markdown("---")
            st.markdown(f"### 🎯 Predicted Car Price: **${prediction:,.2f}**")
            
            # Price category feedback
            if prediction >= 40000:
                st.success("🏎️ Premium vehicle category!")
            elif prediction >= 25000:
                st.info("🚗 Mid-range vehicle category.")
            elif prediction >= 15000:
                st.info("🚙 Budget-friendly category.")
            else:
                st.warning("💡 Economy/older vehicle category.")

