"""
Utility features for ML Portfolio
Contains: search, breadcrumbs, dataset visualization, export, glossary, etc.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
import io
from glossary import GLOSSARY_TERMS


# --- Breadcrumb Navigation ---
def render_breadcrumb(category: str, algorithm: str = None):
    """Render breadcrumb navigation showing current location"""
    parts = ["🏠 Home"]
    
    if category and category != "🏠 Home":
        clean_cat = category.split(' ', 1)[1] if ' ' in category else category
        parts.append(f"📁 {clean_cat}")
    
    if algorithm and algorithm != "-- Select --":
        parts.append(f"📄 {algorithm}")
    
    breadcrumb_html = " → ".join([f'<span class="breadcrumb-item">{p}</span>' for p in parts])
    
    st.markdown(f"""
    <div class="breadcrumb-container">
        {breadcrumb_html}
    </div>
    """, unsafe_allow_html=True)


# --- Search ---
def render_search_box(algo_details: dict, categories: dict):
    """Filter algorithms based on search query from session state"""
    search_query = st.session_state.get("sidebar_search", "")
    
    if search_query:
        results = []
        query_lower = search_query.lower()
        
        for algo_name, details in algo_details.items():
            if (query_lower in algo_name.lower() or 
                query_lower in details.get("desc", "").lower()):
                # Find which category this algorithm belongs to
                for cat_name, cat_data in categories.items():
                    if algo_name in cat_data.get("algorithms", {}):
                        results.append({
                            "name": algo_name,
                            "category": cat_name,
                            "desc": details.get("desc", ""),
                            "icon": details.get("icon", "📘")
                        })
                        break
        
        return results
    return None


# --- Random Algorithm ---
def get_random_algorithm(categories: dict):
    """Get a random algorithm from available categories"""
    all_algos = []
    
    for cat_name, cat_data in categories.items():
        if cat_name != "🏠 Home":
            for algo_name in cat_data.get("algorithms", {}).keys():
                all_algos.append({"category": cat_name, "algorithm": algo_name})
    
    if all_algos:
        return random.choice(all_algos)
    return None


# --- Dataset Visualization ---
def render_dataset_visualization(df: pd.DataFrame, target_col: str = None):
    """Render comprehensive dataset visualization"""
    st.markdown("### 📊 Dataset Exploration")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    viz_tabs = st.tabs(["📈 Distributions", "🔗 Correlations", "📋 Statistics"])
    
    with viz_tabs[0]:
        # Feature distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select feature to visualize", numeric_cols)
            fig = px.histogram(df, x=selected_col, nbins=30, 
                             title=f"Distribution of {selected_col}",
                             template="plotly_white")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        # Correlation heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            fig = px.imshow(corr, 
                          text_auto=".2f",
                          color_continuous_scale="RdBu_r",
                          title="Feature Correlation Heatmap",
                          template="plotly_white")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation heatmap.")
    
    with viz_tabs[2]:
        # Statistical summary
        st.dataframe(df.describe(), use_container_width=True)
        
        # Class balance for classification
        if target_col and target_col in df.columns:
            st.markdown("#### Target Variable Distribution")
            value_counts = df[target_col].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index,
                        title=f"Class Distribution: {target_col}",
                        template="plotly_white")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)


# --- Export ---
def export_results_csv(results: dict, filename: str = "ml_results.csv"):
    """Create downloadable CSV from results dictionary"""
    # Flatten results for CSV
    flat_results = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str)):
            flat_results[key] = value
        elif isinstance(value, np.ndarray):
            flat_results[key] = str(value.tolist())
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                flat_results[f"{key}_{sub_key}"] = sub_val
    
    df = pd.DataFrame([flat_results])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    return csv_buffer.getvalue()


def render_export_button(results: dict, algo_name: str):
    """Render export results button"""
    if results:
        csv_data = export_results_csv(results, f"{algo_name}_results.csv")
        st.download_button(
            label="📥 Export Results (CSV)",
            data=csv_data,
            file_name=f"{algo_name.lower().replace(' ', '_')}_results.csv",
            mime="text/csv",
            key=f"export_{algo_name}"
        )


# --- Training Timer ---
class TrainingTimer:
    """Context manager for measuring training time"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def get_duration_str(self):
        """Get formatted duration string"""
        if self.duration is None:
            return "N/A"
        if self.duration < 1:
            return f"{self.duration * 1000:.1f} ms"
        elif self.duration < 60:
            return f"{self.duration:.2f} sec"
        else:
            minutes = int(self.duration // 60)
            seconds = self.duration % 60
            return f"{minutes}m {seconds:.1f}s"


def render_training_time(duration_str: str):
    """Display training time with nice formatting"""
    st.markdown(f"""
    <div class="training-time-badge">
        ⏱️ Training completed in <strong>{duration_str}</strong>
    </div>
    """, unsafe_allow_html=True)


# --- Reset Button ---
def render_reset_button(session_keys: list, button_key: str = "reset_model"):
    """Render reset button that clears specified session state keys"""
    if st.button("🔄 Reset Model", key=button_key, type="secondary"):
        for key in session_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ Model reset successfully!")
        st.rerun()


# --- Code Copy/Download ---
def render_code_with_copy(code: str, language: str = "python", filename: str = "code.py"):
    """Render code block with copy and download buttons"""
    st.code(code, language=language)
    
    col1, col2 = st.columns(2)
    with col1:
        # Use format instead of f-string to avoid backslash issues
        # Encode the code to base64 for safe JavaScript handling
        import base64
        code_bytes = code.encode('utf-8')
        code_b64 = base64.b64encode(code_bytes).decode('utf-8')
        
        copy_html = """
        <button class="copy-btn" onclick="navigator.clipboard.writeText(atob('{}')).then(() => alert('Code copied to clipboard!'))">
            📋 Copy to Clipboard
        </button>
        """.format(code_b64)
        st.markdown(copy_html, unsafe_allow_html=True)
    
    with col2:
        st.download_button(
            label="📥 Download Code",
            data=code,
            file_name=filename,
            mime="text/plain"
        )


# --- Glossary Page ---
def render_glossary_page():
    """Render the ML glossary page"""
    st.markdown("# 📖 ML Glossary")
    st.markdown("Quick reference guide for common machine learning terminology.")
    
    # Group by category
    categories = {}
    for term, info in GLOSSARY_TERMS.items():
        cat = info.get("category", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((term, info))
    
    # Search filter
    search = st.text_input("🔍 Search terms...", key="glossary_search")
    
    for cat_name, terms in sorted(categories.items()):
        filtered_terms = [(t, i) for t, i in terms 
                         if not search or search.lower() in t.lower() or search.lower() in i["definition"].lower()]
        
        if filtered_terms:
            st.markdown(f"### {cat_name}")
            for term, info in sorted(filtered_terms):
                with st.expander(f"{info['icon']} **{term}**"):
                    # Render HTML content properly
                    st.markdown(info["definition"], unsafe_allow_html=True)


# --- Model Comparison ---
def render_model_comparison_page():
    """Render the Model Comparison Tool page"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target
    
    st.markdown("# ⚖️ Model Comparison Tool")
    st.markdown("Compare multiple algorithms side-by-side on the same dataset.")
    
    # Task type selection
    task_type = st.radio("**Select Task Type:**", ["Classification", "Regression"], horizontal=True)
    
    # Define available algorithms
    if task_type == "Classification":
        available_algos = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(max_depth=10),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10)
        }
    else:
        available_algos = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "SVR": SVR(),
            "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(max_depth=10),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10)
        }
    
    st.markdown("---")
    
    # Algorithm selection
    st.markdown("### 🎯 Select Algorithms to Compare")
    selected_algos = st.multiselect(
        "Choose 2-3 algorithms:",
        list(available_algos.keys()),
        default=list(available_algos.keys())[:2],
        max_selections=3
    )
    
    if len(selected_algos) < 2:
        st.warning("⚠️ Please select at least 2 algorithms to compare.")
        return
    
    st.markdown("---")
    
    # Dataset selection
    st.markdown("### 📊 Select Dataset")
    data_source = st.radio("Data Source:", ["🎲 Synthetic Data", "📁 Upload Your Data"], horizontal=True)
    
    X, y = None, None
    
    if data_source == "📁 Upload Your Data":
        uploaded_file = st.file_uploader("Upload CSV/Excel file", type=['csv', 'xlsx'])
        if uploaded_file:
            df = load_user_dataset(uploaded_file)
            target_col = st.selectbox("Select Target Column:", df.columns)
            X, y = prepare_features_target(df, target_col)
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head())
        else:
            st.info("👆 Upload a dataset to continue.")
            return
    else:
        n_samples = st.slider("Number of Samples:", 200, 2000, 500)
        if task_type == "Classification":
            df = DatasetGenerator.generate_spam_data(n_samples)
            X = df.drop('is_spam', axis=1)
            y = df['is_spam']
            st.info("📧 Using synthetic spam detection dataset")
        else:
            df = DatasetGenerator.generate_house_prices(n_samples)
            X = df.drop('price', axis=1)
            y = df['price']
            st.info("🏠 Using synthetic house price dataset")
        
        with st.expander("📋 Data Preview"):
            st.dataframe(df.head())
    
    st.markdown("---")
    
    # Run comparison
    if st.button("🚀 Compare Models", type="primary", use_container_width=True):
        with st.spinner("Training models..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            results = []
            
            for algo_name in selected_algos:
                model = available_algos[algo_name]
                
                # Clone model for fresh training
                import copy
                model = copy.deepcopy(available_algos[algo_name])
                
                # Train and time
                start_time = time.time()
                model.fit(X_train_scaled, y_train)
                train_time = time.time() - start_time
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                if task_type == "Classification":
                    results.append({
                        "Algorithm": algo_name,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                        "Training Time (s)": train_time
                    })
                else:
                    results.append({
                        "Algorithm": algo_name,
                        "R² Score": r2_score(y_test, y_pred),
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "MSE": mean_squared_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "Training Time (s)": train_time
                    })
            
            # Store results
            st.session_state['comparison_results'] = results
            st.session_state['comparison_task_type'] = task_type
        
        st.success("✅ Comparison complete!")
    
    # Display results
    if 'comparison_results' in st.session_state:
        results = st.session_state['comparison_results']
        task = st.session_state['comparison_task_type']
        
        st.markdown("---")
        st.markdown("## 📊 Comparison Results")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Style the dataframe
        if task == "Classification":
            # Highlight best values
            st.dataframe(
                results_df.style.highlight_max(
                    subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    color='#d4edda'
                ).highlight_min(
                    subset=['Training Time (s)'],
                    color='#d4edda'
                ).format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1 Score': '{:.4f}',
                    'Training Time (s)': '{:.4f}'
                }),
                use_container_width=True
            )
        else:
            st.dataframe(
                results_df.style.highlight_max(
                    subset=['R² Score'],
                    color='#d4edda'
                ).highlight_min(
                    subset=['MAE', 'MSE', 'RMSE', 'Training Time (s)'],
                    color='#d4edda'
                ).format({
                    'R² Score': '{:.4f}',
                    'MAE': '{:.4f}',
                    'MSE': '{:.4f}',
                    'RMSE': '{:.4f}',
                    'Training Time (s)': '{:.4f}'
                }),
                use_container_width=True
            )
        
        # Visualization
        st.markdown("### 📈 Visual Comparison")
        
        if task == "Classification":
            metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        else:
            metric_cols = ['R² Score']
        
        fig = go.Figure()
        for metric in metric_cols:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['Algorithm'],
                y=results_df[metric],
                text=results_df[metric].round(4),
                textposition='outside'
            ))
        
        fig.update_layout(
            title="Algorithm Performance Comparison",
            xaxis_title="Algorithm",
            yaxis_title="Score",
            barmode='group',
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Training time comparison
        fig_time = go.Figure(go.Bar(
            x=results_df['Algorithm'],
            y=results_df['Training Time (s)'],
            text=results_df['Training Time (s)'].round(4),
            textposition='outside',
            marker_color='#667eea'
        ))
        fig_time.update_layout(
            title="Training Time Comparison",
            xaxis_title="Algorithm",
            yaxis_title="Time (seconds)",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Export results
        st.markdown("### 📥 Export Results")
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Results (CSV)",
            data=csv_data,
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )
