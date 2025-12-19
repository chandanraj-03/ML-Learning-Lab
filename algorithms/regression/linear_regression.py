"""
Linear Regression Demo for ML Portfolio
Project: House Price Prediction
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

import sys
sys.path.append('..')
from utils.visualization import plot_regression_results, plot_residuals, plot_learning_curve
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class LinearRegressionDemo:
    def __init__(self):
        self.explanation = get_explanation('linear_regression')
        self.model = None
        self.scaler = StandardScaler()
        
    def render(self):
        """Render the complete Linear Regression demo page"""
        st.markdown(f"# 📈 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        st.markdown(self.explanation['description'])
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔍 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#linear-regression) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/linear-regression-python-implementation/)")
        with tab2:
            data = self._render_demo()
        
        with tab3:
            self._render_predict()
        
        with tab4:
            self._render_code()
        
        with tab5:
            if 'lr_results' in st.session_state:
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
        
        # Data source selection
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
                    
                    # Column selection
                    st.markdown("### Configure Dataset")
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
                        return None
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    return None
            else:
                st.info("👆 Upload a dataset to continue")
                return None
        else:
            # Generate synthetic data
            st.markdown("### Synthetic House Price Data")
            
            col1, col2 = st.columns(2)
            with col1:
                n_samples = st.slider("Number of Samples", 100, 1000, 500)
            with col2:
                noise_level = st.slider("Noise Level", 1, 50, 10)
            
            df = DatasetGenerator.generate_house_prices(n_samples, noise_level)
            
            # Show data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10))
                st.markdown("**Statistics:**")
                st.dataframe(df.describe())
            
            X = df.drop('price', axis=1)
            y = df['price']
        
        # Model configuration
        st.markdown("### Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        with col2:
            fit_intercept = st.checkbox("Fit Intercept", value=True)
        
        # Train button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training Linear Regression model..."):
                results = self._train_model(X, y, test_size, fit_intercept)
                st.session_state['lr_results'] = results
                st.session_state['lr_data'] = (X, y)
                st.success("✅ Model trained successfully! Check the Results tab.")
        
        return df if data_source == "🎲 Synthetic Data" else None
    
    def _render_predict(self):
        """Render prediction interface for user input"""
        st.markdown("## 🔮 Make Predictions")
        
        if 'lr_results' not in st.session_state or 'lr_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        
        st.markdown("### Enter Feature Values")
        st.info("Enter the house details below to predict its price:")
        
        # Create input fields for each feature
        col1, col2 = st.columns(2)
        
        with col1:
            sqft = st.number_input("Square Feet", min_value=500, max_value=10000, value=2000, 
                                   help="🏠 Total living area (500-10,000 sq ft)")
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3,
                                       help="🛏️ Number of bedrooms (1-10)")
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2,
                                        help="🚿 Number of bathrooms (1-5)")
        
        with col2:
            age = st.number_input("House Age (years)", min_value=0, max_value=100, value=10,
                                  help="📅 Age of the house (0-100 years)")
            location_score = st.slider("Location Score", min_value=1.0, max_value=10.0, value=7.0,
                                       help="📍 Neighborhood quality rating (1=poor, 10=excellent)")
        
        if st.button("🎯 Predict Price", type="primary"):
            # Get model and scaler from session state
            model = st.session_state['lr_model']
            scaler = st.session_state['lr_scaler']
            
            # Create input DataFrame with feature names to avoid sklearn warning
            feature_names = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
            input_data = pd.DataFrame([[sqft, bedrooms, bathrooms, age, location_score]], columns=feature_names)
            
            # Scale input using the same scaler
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            st.markdown("---")
            st.markdown("### 💰 Predicted House Price")
            st.markdown(f"## ${prediction:,.2f}")
            
            # Show feature contribution
            st.markdown("### 📊 Feature Contributions")
            coeffs = st.session_state['lr_results']['coefficients']
            contributions = []
            feature_names = ['sqft', 'bedrooms', 'bathrooms', 'age', 'location_score']
            feature_values = [sqft, bedrooms, bathrooms, age, location_score]
            
            for name, value in zip(feature_names, feature_values):
                if name in coeffs:
                    contrib = coeffs[name] * value
                    contributions.append({'Feature': name, 'Value': value, 'Contribution': contrib})
            
            if contributions:
                contrib_df = pd.DataFrame(contributions)
                st.dataframe(contrib_df, width='stretch')

    def _train_model(self, X, y, test_size, fit_intercept):
        """Train the model and return results"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_train_scaled, y_train)
        
        # Store model and scaler in session state for predictions
        st.session_state['lr_model'] = model
        st.session_state['lr_scaler'] = scaler
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            LinearRegression(fit_intercept=fit_intercept),
            X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'coefficients': dict(zip(X.columns, model.coef_)),
            'intercept': model.intercept_
        }
    
    def _render_results(self):
        """Render results and visualizations"""
        results = st.session_state['lr_results']
        
        # Metrics
        st.markdown("## 📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        train_r2 = r2_score(results['y_train'], results['y_train_pred'])
        test_r2 = r2_score(results['y_test'], results['y_test_pred'])
        test_mse = mean_squared_error(results['y_test'], results['y_test_pred'])
        test_mae = mean_absolute_error(results['y_test'], results['y_test_pred'])
        
        with col1:
            st.metric("Train R²", f"{train_r2:.4f}")
        with col2:
            st.metric("Test R²", f"{test_r2:.4f}")
        with col3:
            st.metric("Test MSE", f"{test_mse:.2f}")
        with col4:
            st.metric("Test MAE", f"{test_mae:.2f}")
        
        # Visualizations
        st.markdown("## 📈 Visualizations")
        
        viz_tabs = st.tabs(["Predictions", "Residuals", "Coefficients", "Learning Curve"])
        
        with viz_tabs[0]:
            fig = plot_regression_results(
                results['y_test'].values,
                results['y_test_pred'],
                "Actual vs Predicted (Test Set)"
            )
            st.plotly_chart(fig, width='stretch')
        
        with viz_tabs[1]:
            fig = plot_residuals(
                results['y_test'].values,
                results['y_test_pred'],
                "Residual Analysis"
            )
            st.plotly_chart(fig, width='stretch')
        
        with viz_tabs[2]:
            # Coefficient visualization
            coef_df = pd.DataFrame({
                'Feature': list(results['coefficients'].keys()),
                'Coefficient': list(results['coefficients'].values())
            }).sort_values('Coefficient', key=abs, ascending=True)
            
            fig = go.Figure(go.Bar(
                x=coef_df['Coefficient'],
                y=coef_df['Feature'],
                orientation='h',
                marker=dict(
                    color=coef_df['Coefficient'],
                    colorscale='RdYlBu',
                    colorbar=dict(title="Value")
                )
            ))
            fig.update_layout(
                title="Feature Coefficients",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch')
            
            st.info(f"**Intercept:** {results['intercept']:.4f}")
        
        with viz_tabs[3]:
            fig = plot_learning_curve(
                results['train_sizes'],
                results['train_scores'],
                results['val_scores'],
                "Learning Curve"
            )
            st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        """Render code implementation"""
        st.markdown("## 💻 Complete Implementation Code")
        
        st.markdown("### 📦 Required Dependencies")
        st.code("pip install numpy pandas scikit-learn matplotlib", language="bash")
        
        st.markdown("### 🐍 Full Python Code")
        
        code = '''# ============================================
# LINEAR REGRESSION - House Price Prediction
# ============================================
# Run: pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. GENERATE SAMPLE DATA (or load your own)
# --------------------------------------------
np.random.seed(42)
n_samples = 500

# Create synthetic house data
sqft = np.random.uniform(800, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.uniform(0, 50, n_samples)
location_score = np.random.uniform(1, 10, n_samples)

# Price formula with noise
price = (50000 + 150 * sqft + 20000 * bedrooms + 15000 * bathrooms 
         - 1000 * age + 30000 * location_score 
         + np.random.normal(0, 10000, n_samples))

# Create DataFrame
df = pd.DataFrame({
    'sqft': sqft,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'location_score': location_score,
    'price': price
})

print("Dataset Shape:", df.shape)
print(df.head())

# --------------------------------------------
# 2. PREPARE FEATURES AND TARGET
# --------------------------------------------
X = df.drop('price', axis=1)  # Features
y = df['price']                # Target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --------------------------------------------
# 3. SCALE FEATURES (recommended)
# --------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# 4. TRAIN THE MODEL
# --------------------------------------------
model = LinearRegression(fit_intercept=True)
model.fit(X_train_scaled, y_train)

# --------------------------------------------
# 5. MAKE PREDICTIONS
# --------------------------------------------
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# --------------------------------------------
# 6. EVALUATE THE MODEL
# --------------------------------------------
print("\\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Train R² Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R² Score:  {r2_score(y_test, y_test_pred):.4f}")
print(f"Test MSE:       {mean_squared_error(y_test, y_test_pred):.2f}")
print(f"Test RMSE:      {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
print(f"Test MAE:       {mean_absolute_error(y_test, y_test_pred):.2f}")

# --------------------------------------------
# 7. FEATURE IMPORTANCE (Coefficients)
# --------------------------------------------
print("\\n" + "="*50)
print("FEATURE COEFFICIENTS")
print("="*50)
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:20s}: {coef:>12.4f}")
print(f"{'Intercept':20s}: {model.intercept_:>12.4f}")

# --------------------------------------------
# 8. VISUALIZATION
# --------------------------------------------
plt.figure(figsize=(10, 5))

# Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted')

# Residuals
plt.subplot(1, 2, 2)
residuals = y_test - y_test_pred
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Distribution')

plt.tight_layout()
plt.savefig('linear_regression_results.png')
plt.show()

print("\\n✅ Results saved to 'linear_regression_results.png'")
'''
        st.code(code, language='python')
        
        # Downloadable code
        st.download_button(
            label="📥 Download Complete Code",
            data=code,
            file_name="linear_regression_complete.py",
            mime="text/plain"
        )

