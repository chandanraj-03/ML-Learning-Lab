"""
Visualization utilities for ML Portfolio
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


def plot_regression_results(y_true, y_pred, title="Regression Results"):
    """Plot actual vs predicted values for regression"""
    fig = go.Figure()
    
    # Scatter plot of actual vs predicted
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode='markers',
        marker=dict(
            size=8,
            color=y_pred - y_true,
            colorscale='RdYlBu',
            showscale=True,
            colorbar=dict(title="Error")
        ),
        name='Predictions'
    ))
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.5)', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_residuals(y_true, y_pred, title="Residual Analysis"):
    """Plot residuals for regression analysis"""
    residuals = y_true - y_pred
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Residuals vs Predicted", "Residual Distribution"))
    
    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers',
                   marker=dict(color='#667eea', opacity=0.6)),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=1)
    
    # Residual histogram
    fig.add_trace(
        go.Histogram(x=residuals, marker_color='#764ba2', opacity=0.7),
        row=1, col=2
    )
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_learning_curve(train_sizes, train_scores, val_scores, title="Learning Curve"):
    """Plot learning curve"""
    fig = go.Figure()
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Training score
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='#667eea'),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean - train_std, (train_mean + train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False
    ))
    
    # Validation score
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='#764ba2'),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_mean - val_std, (val_mean + val_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(118, 75, 162, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Training Examples",
        yaxis_title="Score",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    """Plot confusion matrix as heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='RdYlBu',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """Plot decision boundary for 2D classification"""
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # Contour plot
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        colorscale='RdYlBu',
        opacity=0.4,
        showscale=False
    ))
    
    # Scatter plot of data points
    unique_labels = np.unique(y)
    colors = px.colors.qualitative.Set1[:len(unique_labels)]
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1],
            mode='markers',
            marker=dict(size=10, color=colors[i], line=dict(width=1, color='white')),
            name=f'Class {label}'
        ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_feature_importance(feature_names, importances, title="Feature Importance"):
    """Plot feature importance bar chart"""
    sorted_idx = np.argsort(importances)
    
    fig = go.Figure(go.Bar(
        x=importances[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation='h',
        marker=dict(
            color=importances[sorted_idx],
            colorscale='Viridis'
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_cluster_results(X, labels, centers=None, title="Clustering Results"):
    """Plot clustering results with optional centers"""
    fig = go.Figure()
    
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set1
    
    for i, label in enumerate(unique_labels):
        if label == -1:  # Noise points
            color = 'gray'
            name = 'Noise'
        else:
            color = colors[i % len(colors)]
            name = f'Cluster {label}'
        
        mask = labels == label
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1],
            mode='markers',
            marker=dict(size=8, color=color, opacity=0.7),
            name=name
        ))
    
    if centers is not None:
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1],
            mode='markers',
            marker=dict(size=15, color='white', symbol='x', line=dict(width=2)),
            name='Centroids'
        ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_roc_curve(fpr, tpr, auc_score, title="ROC Curve"):
    """Plot ROC curve"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {auc_score:.3f})',
        line=dict(color='#667eea', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_dendrogram(linkage_matrix, labels=None, title="Hierarchical Clustering Dendrogram"):
    """Create dendrogram plot using scipy"""
    from scipy.cluster.hierarchy import dendrogram
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')
    
    dendrogram(linkage_matrix, labels=labels, ax=ax, 
               leaf_rotation=90, leaf_font_size=10,
               above_threshold_color='#667eea')
    
    ax.set_title(title, color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    
    return fig


def plot_elbow_curve(k_values, inertias, title="Elbow Method"):
    """Plot elbow curve for k-means"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k_values, y=inertias,
        mode='lines+markers',
        marker=dict(size=10, color='#667eea'),
        line=dict(color='#667eea', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_silhouette(X, labels, title="Silhouette Analysis"):
    """Plot silhouette scores for each cluster"""
    from sklearn.metrics import silhouette_samples, silhouette_score
    
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    fig = go.Figure()
    
    y_lower = 10
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set1
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        
        cluster_silhouette_values = sample_silhouette_values[labels == label]
        cluster_silhouette_values.sort()
        
        size_cluster = len(cluster_silhouette_values)
        y_upper = y_lower + size_cluster
        
        fig.add_trace(go.Scatter(
            x=cluster_silhouette_values,
            y=np.arange(y_lower, y_upper),
            mode='lines',
            fill='tozerox',
            name=f'Cluster {label}',
            line=dict(color=colors[i % len(colors)])
        ))
        
        y_lower = y_upper + 10
    
    fig.add_vline(x=silhouette_avg, line_dash="dash", line_color="white",
                  annotation_text=f"Avg: {silhouette_avg:.3f}")
    
    fig.update_layout(
        title=title,
        xaxis_title="Silhouette Coefficient",
        yaxis_title="Cluster",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    return fig


def plot_3d_clusters(X, labels, title="3D Clustering"):
    """Plot 3D clustering results"""
    fig = go.Figure()
    
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set1
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            color = 'gray'
            name = 'Noise'
        else:
            color = colors[i % len(colors)]
            name = f'Cluster {label}'
        
        mask = labels == label
        fig.add_trace(go.Scatter3d(
            x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
            mode='markers',
            marker=dict(size=5, color=color, opacity=0.7),
            name=name
        ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)')
        )
    )
    
    return fig


def plot_polynomial_fit(X, y, y_pred, degree, title="Polynomial Regression Fit"):
    """Plot polynomial regression fit"""
    sorted_idx = np.argsort(X.ravel())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=X.ravel()[sorted_idx], y=y[sorted_idx],
        mode='markers',
        marker=dict(size=8, color='#667eea', opacity=0.6),
        name='Data Points'
    ))
    
    fig.add_trace(go.Scatter(
        x=X.ravel()[sorted_idx], y=y_pred[sorted_idx],
        mode='lines',
        line=dict(color='#764ba2', width=3),
        name=f'Polynomial (degree={degree})'
    ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_coefficient_path(alphas, coefs, feature_names=None, title="Coefficient Path"):
    """Plot regularization path showing how coefficients change with alpha"""
    fig = go.Figure()
    
    n_features = coefs.shape[1]
    colors = px.colors.qualitative.Set2
    
    for i in range(n_features):
        name = feature_names[i] if feature_names else f'Feature {i}'
        fig.add_trace(go.Scatter(
            x=np.log10(alphas), y=coefs[:, i],
            mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="log(Alpha)",
        yaxis_title="Coefficient Value",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
