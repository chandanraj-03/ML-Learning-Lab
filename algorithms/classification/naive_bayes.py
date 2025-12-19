"""
Naive Bayes Demo - Sentiment Analysis
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.visualization import plot_confusion_matrix
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator, load_user_dataset, prepare_features_target


class NaiveBayesDemo:
    def __init__(self):
        self.explanation = get_explanation('naive_bayes')
        
    def render(self):
        st.markdown(f"# 📝 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes) or [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'nb_results' in st.session_state:
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
            n_samples = st.slider("Samples", 200, 1000, 500)
            df = DatasetGenerator.generate_review_data(n_samples)
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            X = df.drop('sentiment', axis=1)
            y = df['sentiment']
        
        nb_type = st.selectbox("Naive Bayes Type", ['Gaussian', 'Multinomial'])
        
        if st.button("🚀 Train Model", type="primary"):
            results = self._train_model(X, y, nb_type)
            st.session_state['nb_results'] = results
            st.success("✅ Done!")
    
    def _train_model(self, X, y, nb_type):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if nb_type == 'Gaussian':
            model = GaussianNB()
        else:
            X_train = np.abs(X_train)
            X_test = np.abs(X_test)
            model = MultinomialNB()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.session_state['nb_model'] = model
        st.session_state['nb_feature_names'] = list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        st.session_state['nb_classes'] = list(np.unique(y))
        st.session_state['nb_type'] = nb_type
        
        return {'y_test': y_test, 'y_pred': y_pred, 'nb_type': nb_type}
    
    def _render_results(self):
        r = st.session_state['nb_results']
        
        st.metric("Accuracy", f"{accuracy_score(r['y_test'], r['y_pred']):.4f}")
        
        cm = confusion_matrix(r['y_test'], r['y_pred'])
        labels = ['Negative', 'Neutral', 'Positive'] if len(np.unique(r['y_test'])) == 3 else [str(i) for i in np.unique(r['y_test'])]
        fig = plot_confusion_matrix(cm, labels, "Confusion Matrix")
        st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
NAIVE BAYES - SENTIMENT ANALYSIS
Complete code for Google Colab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# STEP 1: GENERATE SENTIMENT DATA
# ============================================================
def generate_review_data(n_samples=500, random_state=42):
    np.random.seed(random_state)
    
    # Text features (simulated as word frequencies)
    positive_words = np.random.exponential(1, n_samples)
    negative_words = np.random.exponential(1, n_samples)
    neutral_words = np.random.uniform(0, 2, n_samples)
    exclamations = np.random.poisson(1, n_samples)
    word_count = np.random.uniform(10, 200, n_samples)
    
    # Sentiment based on features
    score = positive_words - negative_words + 0.1 * exclamations
    sentiment = np.where(score > 0.5, 2, np.where(score < -0.5, 0, 1))  # 0=Neg, 1=Neutral, 2=Pos
    
    return pd.DataFrame({
        'positive_words': positive_words,
        'negative_words': negative_words,
        'neutral_words': neutral_words,
        'exclamations': exclamations,
        'word_count': word_count,
        'sentiment': sentiment
    })

df = generate_review_data(500)
print("Dataset Preview:")
print(df.head())
print(f"\\nSentiment distribution:")
print(df['sentiment'].value_counts().sort_index())

# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
X = df.drop('sentiment', axis=1)
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\\nTraining: {len(X_train)}, Testing: {len(X_test)}")

# ============================================================
# STEP 3: COMPARE NAIVE BAYES TYPES
# ============================================================
print("\\n📈 COMPARING NAIVE BAYES TYPES:")
print("-" * 40)

# Gaussian NB (for continuous features)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_acc = accuracy_score(y_test, gnb.predict(X_test))
print(f"Gaussian NB: Accuracy = {gnb_acc:.4f}")

# Multinomial NB (for count data - needs positive values)
X_train_pos = np.abs(X_train)
X_test_pos = np.abs(X_test)
mnb = MultinomialNB()
mnb.fit(X_train_pos, y_train)
mnb_acc = accuracy_score(y_test, mnb.predict(X_test_pos))
print(f"Multinomial NB: Accuracy = {mnb_acc:.4f}")

# Use best model
best_model = gnb if gnb_acc >= mnb_acc else mnb
best_name = "Gaussian" if gnb_acc >= mnb_acc else "Multinomial"
print(f"\\n🏆 Best: {best_name} NB")

# ============================================================
# STEP 4: DETAILED EVALUATION
# ============================================================
y_pred = gnb.predict(X_test)
print("\\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
plt.figure(figsize=(10, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.colorbar()
labels = ['Negative', 'Neutral', 'Positive']
plt.xticks([0, 1, 2], labels)
plt.yticks([0, 1, 2], labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes - Sentiment Classification')
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.show()

print("\\n✅ Naive Bayes model trained!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="naive_bayes_sentiment.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render prediction interface"""
        st.markdown("## 🔮 Make Predictions")
        
        if 'nb_model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the Demo tab!")
            return
        
        st.success("✅ Model is trained and ready for predictions!")
        
        model = st.session_state['nb_model']
        feature_names = st.session_state['nb_feature_names']
        classes = st.session_state['nb_classes']
        
        st.markdown("### Enter Feature Values")
        
        input_values = []
        cols = st.columns(min(3, len(feature_names)))
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                val = st.number_input(f"{feature}", value=0.0, key=f"nb_pred_{i}")
                input_values.append(val)
        
        if st.button("🎯 Predict Class", type="primary"):
            input_data = pd.DataFrame([input_values], columns=feature_names)
            if st.session_state['nb_type'] == 'Multinomial':
                input_data = np.abs(input_data)
            
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            st.markdown("---")
            st.markdown(f"### 🎯 Predicted Class: **{prediction}**")
            
            prob_df = pd.DataFrame({'Class': classes, 'Probability': probabilities})
            st.dataframe(prob_df.sort_values('Probability', ascending=False), use_container_width=True)
