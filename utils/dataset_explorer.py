"""
Dataset Explorer Component for ML Learning Lab
Provides sample datasets, auto-detection, and preview visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data.datasets import (
    SAMPLE_DATASETS, 
    analyze_dataset, 
    get_sample_dataset, 
    get_dataset_as_csv,
    load_user_dataset
)


def render_sample_datasets_section():
    """Render sample datasets section for home page"""
    st.markdown("## 📦 Sample Datasets")
    st.markdown("Download ready-to-use datasets for practicing ML algorithms.")
    
    # Group by task type
    regression_datasets = {k: v for k, v in SAMPLE_DATASETS.items() if v['task'] == 'regression'}
    classification_datasets = {k: v for k, v in SAMPLE_DATASETS.items() if v['task'] == 'classification'}
    clustering_datasets = {k: v for k, v in SAMPLE_DATASETS.items() if v['task'] == 'clustering'}
    
    tabs = st.tabs(["📈 Regression", "🎯 Classification", "🔮 Clustering"])
    
    with tabs[0]:
        _render_dataset_cards(regression_datasets)
    
    with tabs[1]:
        _render_dataset_cards(classification_datasets)
    
    with tabs[2]:
        _render_dataset_cards(clustering_datasets)


def _render_dataset_cards(datasets):
    """Render dataset cards in a grid"""
    if not datasets:
        st.info("No datasets in this category.")
        return
    
    cols = st.columns(2)
    for idx, (key, info) in enumerate(datasets.items()):
        with cols[idx % 2]:
            with st.container():
                st.markdown(f"""
                <div style="background: rgba(79, 70, 229, 0.05); border-radius: 12px; padding: 1rem; margin-bottom: 1rem; border-left: 4px solid #4f46e5;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #1e293b;">{info['name']}</h4>
                    <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.5rem;">{info['description']}</p>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        <span style="background: #e0e7ff; color: #4338ca; padding: 0.2rem 0.5rem; border-radius: 8px; font-size: 0.75rem;">
                            {info['samples']} samples
                        </span>
                        <span style="background: #dcfce7; color: #16a34a; padding: 0.2rem 0.5rem; border-radius: 8px; font-size: 0.75rem;">
                            Target: {info['target'] or 'None'}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Download button
                csv_data = get_dataset_as_csv(key)
                st.download_button(
                    label=f"📥 Download {info['name'].split(' ')[1] if len(info['name'].split(' ')) > 1 else info['name']}",
                    data=csv_data,
                    file_name=f"{key}.csv",
                    mime="text/csv",
                    key=f"download_{key}",
                    use_container_width=True
                )


def render_dataset_upload_with_analysis():
    """Render enhanced dataset upload with auto-detection"""
    st.markdown("### 📁 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your own dataset for analysis"
    )
    
    if uploaded_file is not None:
        try:
            df = load_user_dataset(uploaded_file)
            st.success(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Auto-analyze
            analysis = analyze_dataset(df)
            
            # Show analysis results
            _render_analysis_results(df, analysis)
            
            return df, analysis
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None, None
    
    return None, None


def _render_analysis_results(df, analysis):
    """Render the analysis results with visualizations"""
    
    # Data Quality Overview
    st.markdown("### 📊 Data Analysis")
    
    quality = analysis["data_quality"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{quality['total_rows']:,}")
    with col2:
        st.metric("Columns", quality['total_cols'])
    with col3:
        missing_pct = (quality['missing_total'] / (quality['total_rows'] * quality['total_cols'])) * 100
        st.metric("Missing", f"{missing_pct:.1f}%")
    with col4:
        st.metric("Duplicates", quality['duplicate_rows'])
    
    # Target Suggestion
    if analysis["target_suggestion"]:
        confidence_pct = analysis["target_confidence"] * 100
        st.info(f"""
        🎯 **Suggested Target:** `{analysis["target_suggestion"]}` 
        ({confidence_pct:.0f}% confidence) → **{analysis["task_suggestion"]}** task
        """)
    
    # Column Types
    with st.expander("📋 Column Types", expanded=True):
        col_data = []
        for col, dtype in analysis["column_types"].items():
            missing = analysis["data_quality"]["missing_values"].get(col, 0)
            missing_pct = (missing / quality['total_rows']) * 100
            col_data.append({
                "Column": col,
                "Type": dtype.capitalize(),
                "Unique": df[col].nunique(),
                "Missing": f"{missing_pct:.1f}%"
            })
        
        col_df = pd.DataFrame(col_data)
        st.dataframe(col_df, use_container_width=True, hide_index=True)
    
    # Quick Visualizations
    with st.expander("📈 Quick Visualizations"):
        viz_tabs = st.tabs(["Distributions", "Correlations"])
        
        with viz_tabs[0]:
            numeric_cols = analysis["numeric_cols"]
            if numeric_cols:
                selected_col = st.selectbox("Select column", numeric_cols, key="viz_dist_col")
                fig = px.histogram(df, x=selected_col, nbins=30, template="plotly_white")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns for histogram.")
        
        with viz_tabs[1]:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", template="plotly_white")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need 2+ numeric columns for correlation.")


def render_compact_sample_datasets():
    """Render a compact version of sample datasets for algorithm pages"""
    with st.expander("📦 Use Sample Dataset"):
        dataset_options = {key: info['name'] for key, info in SAMPLE_DATASETS.items()}
        
        selected = st.selectbox(
            "Choose a sample dataset",
            options=list(dataset_options.keys()),
            format_func=lambda x: dataset_options[x]
        )
        
        if selected:
            info = SAMPLE_DATASETS[selected]
            st.caption(f"*{info['description']}*")
            st.caption(f"Task: **{info['task']}** | Target: `{info['target']}`")
            
            if st.button("📥 Load This Dataset", key=f"load_{selected}"):
                df = get_sample_dataset(selected)
                return df, info['target']
    
    return None, None
