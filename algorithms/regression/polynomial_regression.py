"""
Polynomial Regression Demo - Battery Degradation
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_polynomial_fit
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class PolynomialRegressionDemo:
    def __init__(self):
        self.explanation = get_explanation('polynomial_regression')
        
    def render(self):
        st.markdown(f"# 📐 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/polynomial-regression-from-scratch-using-python/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'poly_results' in st.session_state:
                self._render_results()
            else:
                st.info("👆 Run the demo first!")
    
    def _render_theory(self):
        st.markdown("## Theory")
        st.markdown(self.explanation['theory'])
        
        st.markdown("### ⚠️ Degree Selection")
        st.warning("""
        **Underfitting (low degree)**: Model too simple, misses patterns  
        **Overfitting (high degree)**: Model too complex, fits noise  
        Use cross-validation to find optimal degree!
        """)
        
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
                feature_col = st.selectbox("Feature Column (single)", [c for c in df.columns if c != target_col])
                X = df[[feature_col]].values
                y = df[target_col].values
            else:
                return
        else:
            n_samples = st.slider("Samples", 50, 300, 200)
            df = DatasetGenerator.generate_battery_data(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X = df[['cycles']].values
            y = df['capacity'].values
        
        st.markdown("### Model Configuration")
        degree = st.slider("Polynomial Degree", 1, 10, 3, help="Higher = more flexible, risk of overfitting")
        
        # Show degree comparison
        if st.checkbox("Compare multiple degrees"):
            degrees_to_compare = st.multiselect("Select degrees", list(range(1, 11)), default=[1, 2, 3, 5])
        else:
            degrees_to_compare = [degree]
        
        if st.button("🚀 Train Model", type="primary"):
            results = self._train_model(X, y, degrees_to_compare)
            st.session_state['poly_results'] = results
            st.success("✅ Done!")
    
    def _train_model(self, X, y, degrees):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {'degrees': [], 'models': [], 'train_r2': [], 'test_r2': [], 'X': X, 'y': y}
        
        for d in degrees:
            model = make_pipeline(PolynomialFeatures(d), LinearRegression())
            model.fit(X_train, y_train)
            
            results['degrees'].append(d)
            results['models'].append(model)
            results['train_r2'].append(r2_score(y_train, model.predict(X_train)))
            results['test_r2'].append(r2_score(y_test, model.predict(X_test)))
        
        # Store best model for prediction
        best_idx = np.argmax(results['test_r2'])
        st.session_state['poly_best_model'] = results['models'][best_idx]
        st.session_state['poly_best_degree'] = results['degrees'][best_idx]
        
        return results
    
    def _render_results(self):
        r = st.session_state['poly_results']
        
        st.markdown("## 📊 Degree Comparison")
        
        # Metrics table
        metrics_df = pd.DataFrame({
            'Degree': r['degrees'],
            'Train R²': [f"{v:.4f}" for v in r['train_r2']],
            'Test R²': [f"{v:.4f}" for v in r['test_r2']]
        })
        st.dataframe(metrics_df, width='stretch')
        
        # Visualization
        st.markdown("## 📈 Fit Comparison")
        
        fig = go.Figure()
        
        # Data points
        fig.add_trace(go.Scatter(
            x=r['X'].ravel(), y=r['y'],
            mode='markers', name='Data',
            marker=dict(size=6, color='#667eea', opacity=0.5)
        ))
        
        # Fitted curves
        X_plot = np.linspace(r['X'].min(), r['X'].max(), 200).reshape(-1, 1)
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7', '#a29bfe']
        
        for i, (d, model) in enumerate(zip(r['degrees'], r['models'])):
            y_plot = model.predict(X_plot)
            fig.add_trace(go.Scatter(
                x=X_plot.ravel(), y=y_plot,
                mode='lines', name=f'Degree {d}',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="Polynomial Fits Comparison",
            xaxis_title="Cycles", yaxis_title="Battery Capacity (%)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Bias-Variance tradeoff
        st.markdown("## 📉 Bias-Variance Tradeoff")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r['degrees'], y=r['train_r2'], mode='lines+markers', name='Train R²'))
        fig.add_trace(go.Scatter(x=r['degrees'], y=r['test_r2'], mode='lines+markers', name='Test R²'))
        fig.update_layout(
            xaxis_title="Degree", yaxis_title="R² Score",
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
POLYNOMIAL REGRESSION - BATTERY DEGRADATION PREDICTION
Complete code for Google Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# STEP 1: GENERATE BATTERY DEGRADATION DATA
# ============================================================
def generate_battery_data(n_samples=200, random_state=42):
    """Generate synthetic battery degradation data"""
    np.random.seed(random_state)
    cycles = np.sort(np.random.uniform(0, 1000, n_samples))
    capacity = (100 - 0.01 * cycles - 0.00005 * cycles**2 + 
                0.00000001 * cycles**3 + np.random.normal(0, 2, n_samples))
    capacity = np.clip(capacity, 20, 100)
    return pd.DataFrame({'cycles': cycles, 'capacity': capacity})

df = generate_battery_data(n_samples=200)
print("Dataset Preview:")
print(df.head())

# ============================================================
# STEP 2: PREPARE DATA FOR TRAINING
# ============================================================
X = df[['cycles']].values
y = df['capacity'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ============================================================
# STEP 3: TRAIN MODELS WITH DIFFERENT DEGREES
# ============================================================
def train_polynomial_model(X_train, y_train, X_test, y_test, degree):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    return {'model': model, 'degree': degree, 'train_r2': train_r2, 'test_r2': test_r2}

# Compare different polynomial degrees
degrees = [1, 2, 3, 4, 5, 7, 10]
results = [train_polynomial_model(X_train, y_train, X_test, y_test, d) for d in degrees]

print("\\n📈 MODEL COMPARISON:")
print("-" * 50)
for r in results:
    print(f"Degree {r['degree']:2d}: Train R²={r['train_r2']:.4f}, Test R²={r['test_r2']:.4f}")

best = max(results, key=lambda x: x['test_r2'])
print(f"\\n🏆 Best Model: Degree {best['degree']} (Test R² = {best['test_r2']:.4f})")

# ============================================================
# STEP 4: VISUALIZATION
# ============================================================
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7']

plt.figure(figsize=(14, 5))

# Plot 1: Data and Polynomial Fits
plt.subplot(1, 2, 1)
plt.scatter(df['cycles'], df['capacity'], alpha=0.5, c='#667eea', s=30, label='Data')
for i, r in enumerate(results[:5]):
    y_plot = r['model'].predict(X_plot)
    plt.plot(X_plot, y_plot, color=colors[i], linewidth=2, 
             label=f"Degree {r['degree']} (R²={r['test_r2']:.3f})")
plt.xlabel('Charging Cycles')
plt.ylabel('Battery Capacity (%)')
plt.title('Polynomial Regression: Battery Degradation')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)

# Plot 2: Bias-Variance Tradeoff
plt.subplot(1, 2, 2)
plt.plot([r['degree'] for r in results], [r['train_r2'] for r in results], 
         'o-', color='#4ecdc4', linewidth=2, markersize=8, label='Train R²')
plt.plot([r['degree'] for r in results], [r['test_r2'] for r in results], 
         's-', color='#ff6b6b', linewidth=2, markersize=8, label='Test R²')
plt.axvspan(5, 10, alpha=0.1, color='red', label='Overfitting Zone')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# STEP 5: MAKE PREDICTIONS
# ============================================================
print("\\n🔮 PREDICTIONS with Best Model:")
print("-" * 40)
best_model = best['model']
for cycles in [0, 100, 250, 500, 750, 1000]:
    pred = best_model.predict([[cycles]])[0]
    print(f"After {cycles:4d} cycles: {pred:.2f}% capacity")

print("\\n✅ Model trained and ready for predictions!")
'''
        st.code(code, language='python')
        
        # Download button
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="polynomial_regression_battery_degradation.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Make Predictions")
        
        if 'poly_best_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        model = st.session_state['poly_best_model']
        degree = st.session_state['poly_best_degree']
        
        st.success(f"✅ Model (degree {degree}) is ready for predictions!")
        
        st.markdown("### Enter Value")
        x_value = st.number_input("Input Value (e.g., cycles)", value=0.0, key="poly_pred")
        
        if st.button("🎯 Predict", type="primary"):
            prediction = model.predict([[x_value]])[0]
            
            st.markdown("---")
            st.markdown(f"### 🎯 Predicted Value: **{prediction:.2f}**")
