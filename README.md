<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">🧠 ML Learning Lab</h1>

<p align="center">
  <strong>An interactive, hands-on machine learning education platform</strong><br>
  <em>Learn ML algorithms through experimentation, not just theory</em>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-algorithms">Algorithms</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 📖 Overview

**ML Learning Lab** is a modern, interactive learning platform built with Streamlit that transforms machine learning from abstract mathematics into something you can **see, tweak, and truly understand**. Instead of just reading theory, you experiment with models, observe their behavior in real-time, and connect intuition with equations.

The platform features **23+ machine learning algorithms** organized across **5 major categories**, guiding you from foundational concepts to practical, production-ready implementations. Each algorithm includes:
- 💻**Streamlit UI**: https://ml-learning-lab-chandan.streamlit.app/
- 📚 **Clear explanations** of theory and mathematical foundations
- 🎮 **Interactive demos** with adjustable parameters
- 📊 **Real-time visualizations** showing how algorithms work
- 💻 **Downloadable Python code** for your own projects

---

## ✨ Features

### 🎯 Core Features

| Feature | Description |
|---------|-------------|
| **Interactive Demos** | Train models, adjust hyperparameters, and instantly see how predictions change |
| **Real-time Visualizations** | Dynamic plots and animations that reveal algorithmic behavior |
| **Structured Learning Paths** | Algorithms grouped by category for progressive skill building |
| **Theory + Practice** | Concise explanations paired with working implementations |
| **Code Downloads** | Production-ready Python code you can use in your projects |
| **ML Glossary** | Comprehensive glossary of ML terms and concepts |
| **Model Comparison** | Compare multiple algorithms side-by-side |
| **Sample Datasets** | Pre-loaded datasets for quick experimentation |

### 🎨 Modern UI/UX

- **Glassmorphism design** with smooth gradients and animations
- **Responsive layout** that works on different screen sizes
- **Dark/Light theme** optimized styling
- **Intuitive navigation** with search and quick actions
- **Difficulty badges** (Beginner → Expert) for each algorithm

---

## 🤖 Algorithms

### 📈 Regression (6 algorithms)
Predict continuous numerical values using various regression techniques.

| Algorithm | Difficulty | Description |
|-----------|------------|-------------|
| Linear Regression | Beginner | Fit a straight line using least squares |
| Polynomial Regression | Intermediate | Capture non-linear relationships with polynomial terms |
| Ridge Regression | Intermediate | L2 regularization to prevent overfitting |
| Lasso Regression | Intermediate | L1 regularization with automatic feature selection |
| Elastic Net | Intermediate | Combines L1 and L2 regularization benefits |
| Support Vector Regression | Advanced | Kernel-based regression for complex patterns |

### 🎯 Classification (5 algorithms)
Assign categorical labels to data points using various classifiers.

| Algorithm | Difficulty | Description |
|-----------|------------|-------------|
| Logistic Regression | Beginner | Binary classification using sigmoid function |
| K-Nearest Neighbors | Beginner | Classify based on closest training examples |
| Support Vector Machine | Advanced | Find optimal hyperplane for classification |
| Naive Bayes | Beginner | Probabilistic classifier using Bayes theorem |
| Perceptron | Beginner | Single-layer neural network for linear boundaries |

### 🌳 Tree-Based (3 algorithms)
Hierarchical decision-making models with interpretable rules.

| Algorithm | Difficulty | Description |
|-----------|------------|-------------|
| Decision Tree | Beginner | Flowchart-like splits for classification/regression |
| Random Forest | Intermediate | Ensemble of decision trees with bagging |
| Gradient Boosting | Advanced | Sequential ensemble with gradient descent |

### 🔮 Unsupervised Learning (5 algorithms)
Discover hidden patterns and structures in unlabeled data.

| Algorithm | Difficulty | Description |
|-----------|------------|-------------|
| K-Means Clustering | Beginner | Partition data into K distinct clusters |
| DBSCAN | Intermediate | Density-based clustering with noise detection |
| Hierarchical Clustering | Intermediate | Build nested cluster hierarchy (dendrogram) |
| Gaussian Mixture Model | Advanced | Soft clustering with probability distributions |
| OPTICS | Advanced | Ordering points to identify cluster structure |

### 🤖 Reinforcement Learning (4 algorithms)
Learn optimal behavior through trial-and-error interaction with environments.

| Algorithm | Difficulty | Description |
|-----------|------------|-------------|
| Q-Learning | Intermediate | Value-based RL with Q-table lookup |
| Deep Q-Network (DQN) | Advanced | Neural network for Q-value approximation |
| REINFORCE | Advanced | Policy gradient with Monte Carlo sampling |
| Multi-Agent RL | Expert | Multiple agents learning cooperatively/competitively |

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/chandanraj-03/ML-Learning-Lab.git
   cd ML-Learning-Lab
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

---

## 📦 Dependencies

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Core** | streamlit | ≥1.28.0 | Web application framework |
| | numpy | ≥1.24.0 | Numerical computing |
| | pandas | ≥2.0.0 | Data manipulation |
| **ML** | scikit-learn | ≥1.3.0 | Machine learning algorithms |
| | xgboost | ≥2.0.0 | Gradient boosting |
| | lightgbm | ≥4.0.0 | Light gradient boosting |
| | catboost | ≥1.2.0 | Categorical boosting |
| **Visualization** | matplotlib | ≥3.7.0 | Static plotting |
| | seaborn | ≥0.12.0 | Statistical visualization |
| | plotly | ≥5.15.0 | Interactive plots |
| **Deep Learning & RL** | torch | ≥2.0.0 | Neural networks |
| | gymnasium | ≥0.29.0 | RL environments |
| **Utilities** | Pillow | ≥10.0.0 | Image processing |

---

## 💻 Usage

### Navigation

1. **Sidebar**: Browse algorithm categories and select specific algorithms
2. **Search**: Use the search box to find algorithms quickly
3. **Surprise Me**: Click for a random algorithm recommendation

### Algorithm Pages

Each algorithm page has three main tabs:

| Tab | Content |
|-----|---------|
| **📚 Learn** | Theory, mathematical foundations, use cases, and key concepts |
| **🎮 Demo** | Interactive visualization with adjustable parameters |
| **💻 Code** | Downloadable Python implementation |

### Example Workflow

1. Select **📈 Regression** from the sidebar
2. Choose **Linear Regression** from the algorithm dropdown
3. Read the theory in the **Learn** tab
4. Experiment with parameters in the **Demo** tab
5. Download the code from the **Code** tab

---

## 📁 Project Structure

```
ML-Learning-Lab/
├── 📄 app.py                    # Main Streamlit application
├── 📄 styles.css                # Custom CSS styling
├── 📄 glossary.py               # ML glossary terms and definitions
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # This file
│
├── 📂 algorithms/               # Algorithm implementations
│   ├── 📂 regression/           # Regression algorithms
│   │   ├── linear_regression.py
│   │   ├── polynomial_regression.py
│   │   ├── ridge_regression.py
│   │   ├── lasso_regression.py
│   │   ├── elastic_net.py
│   │   └── svr.py
│   │
│   ├── 📂 classification/       # Classification algorithms
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   ├── svm.py
│   │   ├── naive_bayes.py
│   │   └── perceptron.py
│   │
│   ├── 📂 tree_based/           # Tree-based algorithms
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   └── gradient_boosting.py
│   │
│   ├── 📂 unsupervised/         # Clustering algorithms
│   │   ├── kmeans.py
│   │   ├── dbscan.py
│   │   ├── hierarchical.py
│   │   ├── gmm.py
│   │   └── optics.py
│   │
│   └── 📂 reinforcement/        # RL algorithms
│       ├── q_learning.py
│       ├── dqn.py
│       ├── reinforce.py
│       └── marl.py
│
├── 📂 utils/                    # Utility modules
│   ├── features.py              # UI components and helpers
│   ├── visualization.py         # Plotting utilities
│   ├── explanations.py          # Algorithm explanations
│   └── dataset_explorer.py      # Sample dataset handling
│
├── 📂 data/                     # Data handling
│   └── datasets.py              # Dataset generation and loading
│
└── 📂 colab_notebooks/          # Jupyter/Colab notebooks
    └── polynomial_regression_battery_degradation.py
```

---

## 🎓 Learning Path

### For Beginners
Start with these algorithms to build foundational understanding:

1. **Linear Regression** → Understand basic prediction
2. **Logistic Regression** → Learn classification basics
3. **K-Nearest Neighbors** → Intuitive distance-based learning
4. **Decision Trees** → Interpretable rule-based models
5. **K-Means** → Introduction to clustering

### For Intermediate Learners
Progress to these algorithms:

1. **Ridge/Lasso Regression** → Regularization concepts
2. **Random Forest** → Ensemble methods
3. **DBSCAN** → Density-based clustering
4. **Q-Learning** → Reinforcement learning basics

### For Advanced Learners
Challenge yourself with:

1. **Support Vector Machines** → Kernel methods
2. **Gradient Boosting** → Sequential ensembles
3. **Deep Q-Network** → Deep reinforcement learning
4. **Multi-Agent RL** → Complex agent interactions

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

- 🐛 **Bug Reports**: Found an issue? Open a GitHub issue
- ✨ **Feature Requests**: Have ideas? Share them with us
- 📝 **Documentation**: Help improve explanations
- 🧪 **New Algorithms**: Add more ML algorithms
- 🎨 **UI Improvements**: Enhance the user experience

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Chandan Raj**

- GitHub: [@chandanraj-03](https://github.com/chandanraj-03)

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Scikit-learn](https://scikit-learn.org/) for ML implementations
- [Plotly](https://plotly.com/) for interactive visualizations
- The open-source ML community for inspiration

---

<p align="center">
  <strong>⭐ Star this repository if you find it helpful! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ for the ML learning community
</p>
