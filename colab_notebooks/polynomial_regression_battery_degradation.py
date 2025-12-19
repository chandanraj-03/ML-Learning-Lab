"""
================================================================================
POLYNOMIAL REGRESSION - BATTERY DEGRADATION PREDICTION
================================================================================
A complete, self-contained implementation for Google Colab.

This notebook demonstrates:
- Polynomial Regression for modeling battery capacity degradation
- Comparison of different polynomial degrees
- Bias-Variance tradeoff visualization
- Prediction capabilities

Run this in Google Colab: https://colab.research.google.com/
================================================================================
"""

# ============================================================================
# STEP 1: INSTALL AND IMPORT DEPENDENCIES
# ============================================================================
# Uncomment the line below if running in Colab and plotly is not installed
# !pip install plotly -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Try importing plotly for interactive plots, fall back to matplotlib
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    USE_PLOTLY = True
except ImportError:
    USE_PLOTLY = False
    print("📊 Plotly not installed. Using Matplotlib for visualizations.")
    print("   Install with: !pip install plotly")

print("✅ All dependencies loaded successfully!")

# ============================================================================
# STEP 2: GENERATE SYNTHETIC BATTERY DEGRADATION DATA
# ============================================================================
def generate_battery_data(n_samples=200, random_state=42):
    """
    Generate synthetic battery degradation data for Polynomial Regression.
    
    Battery capacity follows a polynomial decay pattern:
    - Initial capacity: 100%
    - Degrades with charging cycles following polynomial curve
    - Includes realistic noise
    
    Parameters:
    -----------
    n_samples : int
        Number of data points to generate
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame with columns: 'cycles', 'capacity'
    """
    np.random.seed(random_state)
    
    # Charging cycles (0 to 1000)
    cycles = np.sort(np.random.uniform(0, 1000, n_samples))
    
    # Battery capacity follows polynomial decay
    # Real batteries show non-linear degradation patterns
    capacity = (
        100 -                           # Initial capacity 100%
        0.01 * cycles -                 # Linear degradation component
        0.00005 * cycles**2 +           # Quadratic decay
        0.00000001 * cycles**3 +        # Cubic component
        np.random.normal(0, 2, n_samples)  # Measurement noise
    )
    
    # Clip capacity to realistic range (20% - 100%)
    capacity = np.clip(capacity, 20, 100)
    
    return pd.DataFrame({
        'cycles': cycles,
        'capacity': capacity
    })

# Generate data
print("\n" + "="*60)
print("📊 GENERATING BATTERY DEGRADATION DATA")
print("="*60)

df = generate_battery_data(n_samples=200)
print(f"\n📈 Dataset Shape: {df.shape}")
print(f"📉 Cycles Range: {df['cycles'].min():.0f} - {df['cycles'].max():.0f}")
print(f"🔋 Capacity Range: {df['capacity'].min():.1f}% - {df['capacity'].max():.1f}%")
print("\n📋 First 10 rows:")
print(df.head(10).to_string(index=False))

# ============================================================================
# STEP 3: DATA VISUALIZATION
# ============================================================================
print("\n" + "="*60)
print("📊 DATA VISUALIZATION")
print("="*60)

if USE_PLOTLY:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['cycles'], 
        y=df['capacity'],
        mode='markers',
        name='Battery Data',
        marker=dict(size=8, color='#667eea', opacity=0.6)
    ))
    fig.update_layout(
        title='Battery Capacity vs Charging Cycles',
        xaxis_title='Charging Cycles',
        yaxis_title='Battery Capacity (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.show()
else:
    plt.figure(figsize=(10, 6))
    plt.scatter(df['cycles'], df['capacity'], alpha=0.6, c='#667eea', s=40)
    plt.xlabel('Charging Cycles')
    plt.ylabel('Battery Capacity (%)')
    plt.title('Battery Capacity vs Charging Cycles')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================================
# STEP 4: PREPARE DATA FOR TRAINING
# ============================================================================
print("\n" + "="*60)
print("🔧 PREPARING DATA FOR TRAINING")
print("="*60)

# Features (X) and Target (y)
X = df[['cycles']].values
y = df['capacity'].values

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📚 Training samples: {len(X_train)}")
print(f"📝 Testing samples: {len(X_test)}")

# ============================================================================
# STEP 5: POLYNOMIAL REGRESSION IMPLEMENTATION
# ============================================================================
print("\n" + "="*60)
print("🚀 POLYNOMIAL REGRESSION - TRAINING MODELS")
print("="*60)

def train_polynomial_model(X_train, y_train, X_test, y_test, degree):
    """
    Train a polynomial regression model of specified degree.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Testing data  
    degree : int - Polynomial degree
    
    Returns:
    --------
    dict with model, predictions, and metrics
    """
    # Create pipeline: PolynomialFeatures -> LinearRegression
    model = make_pipeline(
        PolynomialFeatures(degree, include_bias=True),
        LinearRegression()
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    return {
        'model': model,
        'degree': degree,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae
    }

# Train models with different polynomial degrees
degrees_to_compare = [1, 2, 3, 4, 5, 7, 10]
results = []

print("\n📈 Training polynomial models of different degrees...\n")

for degree in degrees_to_compare:
    result = train_polynomial_model(X_train, y_train, X_test, y_test, degree)
    results.append(result)
    print(f"  Degree {degree:2d}: Train R² = {result['train_r2']:.4f}, Test R² = {result['test_r2']:.4f}")

print("\n✅ All models trained successfully!")

# ============================================================================
# STEP 6: RESULTS COMPARISON TABLE
# ============================================================================
print("\n" + "="*60)
print("📊 MODEL PERFORMANCE COMPARISON")
print("="*60)

# Create comparison DataFrame
comparison_df = pd.DataFrame([
    {
        'Degree': r['degree'],
        'Train R²': f"{r['train_r2']:.4f}",
        'Test R²': f"{r['test_r2']:.4f}",
        'Train MSE': f"{r['train_mse']:.4f}",
        'Test MSE': f"{r['test_mse']:.4f}",
        'Train MAE': f"{r['train_mae']:.4f}",
        'Test MAE': f"{r['test_mae']:.4f}"
    }
    for r in results
])

print("\n" + comparison_df.to_string(index=False))

# Find best model based on test R²
best_result = max(results, key=lambda x: x['test_r2'])
print(f"\n🏆 Best Model: Degree {best_result['degree']} (Test R² = {best_result['test_r2']:.4f})")

# ============================================================================
# STEP 7: VISUALIZE POLYNOMIAL FITS
# ============================================================================
print("\n" + "="*60)
print("📈 VISUALIZING POLYNOMIAL FITS")
print("="*60)

# Generate smooth curve for plotting
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7', '#a29bfe', '#fd79a8']

if USE_PLOTLY:
    fig = go.Figure()
    
    # Plot original data points
    fig.add_trace(go.Scatter(
        x=df['cycles'], 
        y=df['capacity'],
        mode='markers',
        name='Actual Data',
        marker=dict(size=6, color='#667eea', opacity=0.5)
    ))
    
    # Plot fitted curves for each degree
    for i, result in enumerate(results):
        y_plot = result['model'].predict(X_plot)
        fig.add_trace(go.Scatter(
            x=X_plot.ravel(),
            y=y_plot,
            mode='lines',
            name=f'Degree {result["degree"]} (R²={result["test_r2"]:.3f})',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title='Polynomial Regression Fits Comparison',
        xaxis_title='Charging Cycles',
        yaxis_title='Battery Capacity (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(x=0.02, y=0.02, bgcolor='rgba(0,0,0,0.5)')
    )
    fig.show()
else:
    plt.figure(figsize=(12, 7))
    plt.scatter(df['cycles'], df['capacity'], alpha=0.5, c='#667eea', s=30, label='Data')
    
    for i, result in enumerate(results):
        y_plot = result['model'].predict(X_plot)
        plt.plot(X_plot, y_plot, color=colors[i % len(colors)], linewidth=2,
                 label=f'Degree {result["degree"]} (R²={result["test_r2"]:.3f})')
    
    plt.xlabel('Charging Cycles')
    plt.ylabel('Battery Capacity (%)')
    plt.title('Polynomial Regression Fits Comparison')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================================
# STEP 8: BIAS-VARIANCE TRADEOFF VISUALIZATION
# ============================================================================
print("\n" + "="*60)
print("📉 BIAS-VARIANCE TRADEOFF ANALYSIS")
print("="*60)

degrees = [r['degree'] for r in results]
train_r2_scores = [r['train_r2'] for r in results]
test_r2_scores = [r['test_r2'] for r in results]

if USE_PLOTLY:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=degrees, y=train_r2_scores,
        mode='lines+markers',
        name='Train R²',
        line=dict(color='#4ecdc4', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=degrees, y=test_r2_scores,
        mode='lines+markers',
        name='Test R²',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=10)
    ))
    
    # Add annotation for overfitting region
    fig.add_vrect(x0=5, x1=10, fillcolor="red", opacity=0.1,
                  annotation_text="⚠️ Overfitting Risk", annotation_position="top left")
    
    fig.update_layout(
        title='Bias-Variance Tradeoff: Train vs Test R² by Polynomial Degree',
        xaxis_title='Polynomial Degree',
        yaxis_title='R² Score',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.show()
else:
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_r2_scores, 'o-', color='#4ecdc4', linewidth=2, 
             markersize=8, label='Train R²')
    plt.plot(degrees, test_r2_scores, 's-', color='#ff6b6b', linewidth=2, 
             markersize=8, label='Test R²')
    plt.axvspan(5, 10, alpha=0.1, color='red', label='Overfitting Risk Zone')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R² Score')
    plt.title('Bias-Variance Tradeoff: Train vs Test R² by Polynomial Degree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("""
📚 OBSERVATIONS:
---------------
• Low degree (1-2): UNDERFITTING - Model too simple, misses patterns
• Optimal degree (3-4): GOOD FIT - Captures true pattern without overfitting
• High degree (7+): OVERFITTING - Fits training noise, poor generalization

💡 TIP: Use cross-validation to find the optimal degree!
""")

# ============================================================================
# STEP 9: DETAILED BEST MODEL ANALYSIS
# ============================================================================
print("\n" + "="*60)
print(f"🏆 BEST MODEL ANALYSIS (Degree {best_result['degree']})")
print("="*60)

best_model = best_result['model']

# Get coefficients
poly_features = best_model.named_steps['polynomialfeatures']
lin_reg = best_model.named_steps['linearregression']

print("\n📐 Model Equation:")
print("-" * 40)

coefs = lin_reg.coef_
intercept = lin_reg.intercept_

equation_parts = [f"{intercept:.4f}"]
for i, coef in enumerate(coefs[1:], start=1):  # Skip bias term from PolynomialFeatures
    if coef >= 0:
        equation_parts.append(f"+ {coef:.8f}x^{i}")
    else:
        equation_parts.append(f"- {abs(coef):.8f}x^{i}")

print(f"y = {' '.join(equation_parts)}")

print(f"\n📊 Performance Metrics:")
print("-" * 40)
print(f"  Train R²:  {best_result['train_r2']:.4f}")
print(f"  Test R²:   {best_result['test_r2']:.4f}")
print(f"  Train MSE: {best_result['train_mse']:.4f}")
print(f"  Test MSE:  {best_result['test_mse']:.4f}")
print(f"  Train MAE: {best_result['train_mae']:.4f}")
print(f"  Test MAE:  {best_result['test_mae']:.4f}")

# ============================================================================
# STEP 10: MAKE PREDICTIONS
# ============================================================================
print("\n" + "="*60)
print("🔮 MAKING PREDICTIONS")
print("="*60)

def predict_capacity(model, cycles):
    """Predict battery capacity for given number of cycles."""
    return model.predict([[cycles]])[0]

# Example predictions
test_cycles = [0, 100, 250, 500, 750, 1000]

print("\n📱 Predicted Battery Capacity at Different Cycle Counts:")
print("-" * 50)
print(f"{'Cycles':>10} | {'Predicted Capacity':>20}")
print("-" * 50)

for cycles in test_cycles:
    predicted = predict_capacity(best_model, cycles)
    print(f"{cycles:>10} | {predicted:>18.2f}%")

print("-" * 50)

# ============================================================================
# STEP 11: INTERACTIVE PREDICTION (For Colab)
# ============================================================================
print("\n" + "="*60)
print("💡 INTERACTIVE PREDICTION")
print("="*60)

try:
    # Try to get user input (works in Colab with IPython widgets)
    user_cycles = float(input("\n🔢 Enter number of charging cycles to predict (or press Enter for 500): ") or 500)
    prediction = predict_capacity(best_model, user_cycles)
    print(f"\n🔋 Predicted Battery Capacity after {user_cycles:.0f} cycles: {prediction:.2f}%")
except:
    print("\n📝 For interactive prediction, run in Google Colab!")
    prediction = predict_capacity(best_model, 500)
    print(f"🔋 Example: Predicted capacity after 500 cycles: {prediction:.2f}%")

# ============================================================================
# STEP 12: RESIDUAL ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("📊 RESIDUAL ANALYSIS")
print("="*60)

y_test_pred = best_model.predict(X_test)
residuals = y_test - y_test_pred

if USE_PLOTLY:
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Actual vs Predicted', 'Residuals Distribution'))
    
    # Actual vs Predicted
    fig.add_trace(go.Scatter(
        x=y_test, y=y_test_pred,
        mode='markers',
        name='Predictions',
        marker=dict(size=8, color='#667eea', opacity=0.6)
    ), row=1, col=1)
    
    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ), row=1, col=1)
    
    # Residuals histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        name='Residuals',
        marker_color='#4ecdc4',
        nbinsx=20
    ), row=1, col=2)
    
    fig.update_layout(
        title=f'Residual Analysis (Degree {best_result["degree"]} Model)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    fig.update_xaxes(title_text='Actual Capacity', row=1, col=1)
    fig.update_yaxes(title_text='Predicted Capacity', row=1, col=1)
    fig.update_xaxes(title_text='Residual Value', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)
    fig.show()
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_test, y_test_pred, alpha=0.6, c='#667eea', s=40)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Capacity')
    axes[0].set_ylabel('Predicted Capacity')
    axes[0].set_title('Actual vs Predicted')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(residuals, bins=20, color='#4ecdc4', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Residual Analysis (Degree {best_result["degree"]} Model)')
    plt.tight_layout()
    plt.show()

print(f"""
📊 RESIDUAL STATISTICS:
-----------------------
  Mean Residual:  {np.mean(residuals):.4f} (should be ~0)
  Std Residual:   {np.std(residuals):.4f}
  Min Residual:   {np.min(residuals):.4f}
  Max Residual:   {np.max(residuals):.4f}
""")

# ============================================================================
# STEP 13: SAVE MODEL (Optional)
# ============================================================================
print("\n" + "="*60)
print("💾 SAVING MODEL")
print("="*60)

try:
    import joblib
    model_filename = 'polynomial_regression_battery_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"\n✅ Model saved as '{model_filename}'")
    print("   To load: model = joblib.load('polynomial_regression_battery_model.pkl')")
except Exception as e:
    print(f"\n⚠️ Could not save model: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("📋 SUMMARY")
print("="*60)

print(f"""
🔋 BATTERY DEGRADATION PREDICTION - POLYNOMIAL REGRESSION

📊 Dataset:
   - {len(df)} samples of battery capacity measurements
   - Feature: Charging cycles (0-1000)
   - Target: Battery capacity (%)

🏆 Best Model: Polynomial Degree {best_result['degree']}
   - Test R² Score: {best_result['test_r2']:.4f}
   - Test MSE: {best_result['test_mse']:.4f}
   - Test MAE: {best_result['test_mae']:.4f}

📈 Key Insights:
   1. Battery capacity degrades non-linearly with cycles
   2. Polynomial regression captures this degradation pattern
   3. Higher degrees risk overfitting - choose wisely!
   4. Cross-validation helps find optimal complexity

🔮 Sample Predictions:
   - After 0 cycles: ~100% capacity
   - After 500 cycles: ~{predict_capacity(best_model, 500):.1f}% capacity
   - After 1000 cycles: ~{predict_capacity(best_model, 1000):.1f}% capacity

✅ Model is trained and ready for predictions!
""")

print("="*60)
print("🎉 COMPLETED! Happy Machine Learning!")
print("="*60)
