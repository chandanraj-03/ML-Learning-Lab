"""
ML Learning Portfolio - Interactive Machine Learning Education
A modern Streamlit-based learning platform for ML algorithms.
"""
import streamlit as st
import random

st.set_page_config(
    page_title="ML Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    with open("styles.css", "r") as f:
        return f"<style>{f.read()}</style>"
st.markdown(load_css(), unsafe_allow_html=True)

st.markdown("""
<div class="ambient-glow glow-1"></div>
<div class="ambient-glow glow-2"></div>
<div class="ambient-glow glow-3"></div>
""", unsafe_allow_html=True)

from utils.features import (
    render_breadcrumb, render_search_box, get_random_algorithm,
    render_glossary_page, render_model_comparison_page, GLOSSARY_TERMS
)

from algorithms.regression import (
    LinearRegressionDemo, LassoRegressionDemo, RidgeRegressionDemo,
    ElasticNetDemo, SVRDemo, PolynomialRegressionDemo
)
from algorithms.classification import (
    LogisticRegressionDemo, KNNDemo, SVMDemo, NaiveBayesDemo, PerceptronDemo
)
from algorithms.tree_based import (
    DecisionTreeDemo, RandomForestDemo, GradientBoostingDemo
)
from algorithms.unsupervised import (
    KMeansDemo, DBSCANDemo, HierarchicalDemo, GMMDemo, OPTICSDemo
)
from algorithms.reinforcement import (
    QLearningDemo, DQNDemo, REINFORCEDemo, MARLDemo
)

ALGO_DETAILS = {
    "Linear Regression": {"desc": "Predict continuous values using a straight line", "difficulty": "Beginner", "icon": "📊"},
    "Polynomial Regression": {"desc": "Fit curves to capture non-linear relationships", "difficulty": "Intermediate", "icon": "📈"},
    "Ridge Regression": {"desc": "Linear regression with L2 regularization", "difficulty": "Intermediate", "icon": "🎯"},
    "Lasso Regression": {"desc": "Feature selection with L1 regularization", "difficulty": "Intermediate", "icon": "✂️"},
    "Elastic Net": {"desc": "Combines L1 and L2 regularization", "difficulty": "Intermediate", "icon": "🔗"},
    "Support Vector Regression": {"desc": "Kernel-based regression for complex patterns", "difficulty": "Advanced", "icon": "🎪"},
    "Logistic Regression": {"desc": "Binary classification using sigmoid function", "difficulty": "Beginner", "icon": "🎲"},
    "K-Nearest Neighbors": {"desc": "Classify based on closest training examples", "difficulty": "Beginner", "icon": "👥"},
    "Support Vector Machine": {"desc": "Find optimal hyperplane for classification", "difficulty": "Advanced", "icon": "⚔️"},
    "Naive Bayes": {"desc": "Probabilistic classifier using Bayes theorem", "difficulty": "Beginner", "icon": "🎰"},
    "Perceptron": {"desc": "Single-layer neural network for linear boundaries", "difficulty": "Beginner", "icon": "🧠"},
    "Decision Tree": {"desc": "Tree-based splits for classification/regression", "difficulty": "Beginner", "icon": "🌲"},
    "Random Forest": {"desc": "Ensemble of decision trees with bagging", "difficulty": "Intermediate", "icon": "🌳"},
    "Gradient Boosting": {"desc": "Sequential ensemble with gradient descent", "difficulty": "Advanced", "icon": "🚀"},
    "K-Means Clustering": {"desc": "Partition data into K distinct clusters", "difficulty": "Beginner", "icon": "⭕"},
    "DBSCAN": {"desc": "Density-based clustering with noise detection", "difficulty": "Intermediate", "icon": "🔍"},
    "Hierarchical Clustering": {"desc": "Build nested cluster hierarchy", "difficulty": "Intermediate", "icon": "📊"},
    "Gaussian Mixture Model": {"desc": "Soft clustering with probability distributions", "difficulty": "Advanced", "icon": "🎨"},
    "OPTICS": {"desc": "Ordering points for cluster structure", "difficulty": "Advanced", "icon": "📡"},
    "Q-Learning": {"desc": "Value-based RL with Q-table lookup", "difficulty": "Intermediate", "icon": "🎮"},
    "Deep Q-Network (DQN)": {"desc": "Neural networks for Q-value approximation", "difficulty": "Advanced", "icon": "🤖"},
    "REINFORCE": {"desc": "Policy gradient with Monte Carlo sampling", "difficulty": "Advanced", "icon": "📰"},
    "Multi-Agent RL": {"desc": "Multiple agents learning together", "difficulty": "Expert", "icon": "🤝"},
}

CATEGORY_EXPLANATIONS = {
    "📈 Regression": {
        "overview": "Regression algorithms are supervised learning techniques used to predict continuous numerical values. They learn the relationship between input features (independent variables) and a target variable (dependent variable) by fitting a mathematical function to labeled data. Once trained, the model can estimate outcomes for new, unseen inputs—things like predicting house prices, temperatures, sales, or scores.<br><br> At their core, regression models are curve-fitters with discipline: they don’t just memorize data, they try to capture the underlying pattern that generates it.",
        "use_cases": ["House price prediction", "Sales forecasting", "Stock price analysis", "Weather prediction", "Demand estimation"],
        "types": {
            "Linear Regression": {
                "description": "It is the most basic regression algorithm and fits a straight line of the form y = mx + b through the data by minimizing the sum of squared errors (also called least squares). The idea is to choose the line where the squared vertical distances between the actual data points and the predicted values are as small as possible.<br><br>This simplicity is its strength: it’s easy to interpret, fast to train, and often surprisingly effective—until the real world decides to be nonlinear.",
                "when_to_use": "Use when the relationship between features and target is approximately linear and you need an interpretable model.",
                "key_features": ["Simple and fast", "Highly interpretable", "Works well with linear relationships", "Sensitive to outliers"],
                "formula": "y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ"
            },
            "Polynomial Regression": {
                "description": "Polynomial regression extends linear regression by transforming the input features into polynomial terms (such as 𝑥<sup>2</sup>, x<sup>3</sup>, etc.), allowing the model to capture non-linear relationships between variables. Despite modeling curves, it is still a linear model at heart because it remains linear with respect to the coefficients.<br><br>In short: the equation bends, the math stays linear, and the model gets better at following curved patterns—up to the point where it starts overfitting if you let the degree grow wild.",
                "when_to_use": "Use when the relationship between features and target is curved or non-linear but you want to keep using least squares approach.",
                "key_features": ["Captures non-linear patterns", "Flexible degree selection", "Can overfit with high degrees", "Good for single-variable curves"],
                "formula": "y = β₀ + β₁x + β₂x² + ... + βₙxⁿ"
            },
            "Ridge Regression": {
                "description": "Ridge regression extends linear regression by adding L2 regularization, which penalizes large coefficient values by adding the sum of their squares to the loss function. This discourages overly complex models, reduces variance, and helps prevent overfitting—especially when features are highly correlated.<br><br>Think of it as linear regression wearing a seatbelt: it still goes where the data leads, but it won’t swerve wildly to fit every noise bump along the road.",
                "when_to_use": "Use when you have many features, multicollinearity issues, or want to prevent overfitting while keeping all features.",
                "key_features": ["Prevents overfitting", "Handles multicollinearity", "Keeps all features", "Shrinks coefficients toward zero"],
                "formula": "Minimize: Σ(yᵢ - ŷᵢ)² + λΣβⱼ²"
            },
            "Lasso Regression": {
                "description": "Lasso Regression (Least Absolute Shrinkage and Selection Operator).<br>Lasso regression applies L1 regularization, which penalizes the absolute values of the coefficients. This has a special side effect: some coefficients can be driven exactly to zero, effectively removing those features from the model. As a result, Lasso performs automatic feature selection while still doing regression.<br><br>In essence, Ridge spreads the weight thin; Lasso snaps weak links entirely—useful when you suspect only a few features really matter.",
                "when_to_use": "Use when you want automatic feature selection and suspect many features are irrelevant.",
                "key_features": ["Automatic feature selection", "Creates sparse models", "Handles high-dimensional data", "More interpretable output"],
                "formula": "Minimize: Σ(yᵢ - ŷᵢ)² + λΣ|βⱼ|"
            },
            "Elastic Net": {
                "description": "Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization into a single model. This balance allows it to perform feature selection like Lasso while also shrinking coefficients and handling multicollinearity like Ridge.<br><br>It’s the diplomatic middle ground of linear models: not as ruthless as Lasso, not as cautious as Ridge, and often the best choice when you have many correlated features and want both stability and sparsity.",
                "when_to_use": "Use when you have correlated features and want both regularization and some feature selection.",
                "key_features": ["Combines L1 and L2 benefits", "Handles correlated features", "Flexible mixing parameter", "Robust to group correlations"],
                "formula": "Minimize: Σ(yᵢ - ŷᵢ)² + λ₁Σ|βⱼ| + λ₂Σβⱼ²"
            },
            "Support Vector Regression": {
                "description": "SVR uses kernel functions to implicitly map input data into higher-dimensional spaces, where a linear relationship may exist. It then finds a hyperplane that fits the data while keeping prediction errors within a specified margin of tolerance (the ε-insensitive zone). Errors inside this margin are ignored; errors outside are penalized.<br><br>The clever trick is that the model focuses on boundary points—support vectors—rather than every data point. It’s regression with a geometric spine and a tolerance for small mistakes, which makes it powerful for complex, non-linear patterns.",
                "when_to_use": "Use for complex non-linear relationships and when you need robust performance with outliers.",
                "key_features": ["Handles non-linear patterns", "Uses kernel trick", "Robust to outliers", "Good for small-medium datasets"],
                "formula": "Minimize: ½||w||² + CΣ(ξᵢ + ξᵢ*)"
            }
        }
    },
    "🎯 Classification": {
        "overview": "Classification algorithms are supervised learning methods that predict categorical labels or classes. They learn decision boundaries from labeled training data, mapping input features to discrete output categories. Once trained, these models can assign new, unseen instances to one of the predefined classes—such as spam vs. not spam, disease vs. healthy, or fraud vs. legitimate.<br><br>Where regression draws a line to predict a number, classification draws a border to decide a label. Different goal, same learning machinery.",
        "use_cases": ["Email spam detection", "Medical diagnosis", "Customer churn prediction", "Image recognition", "Sentiment analysis"],
        "types": {
            "Logistic Regression": {
                "description": "Despite the misleading name, logistic regression is a classification algorithm, not a regression one. It uses the logistic (sigmoid) function to map a linear combination of input features to a value between 0 and 1, which is interpreted as the probability of a binary outcome. A threshold (often 0.5) then converts that probability into a class label.<br><br>So it doesn’t predict numbers like prices—it predicts confidence. The “regression” part is historical baggage, not a job description.",
                "when_to_use": "Use for binary classification when you need probability outputs and interpretable coefficients.",
                "key_features": ["Outputs probabilities", "Highly interpretable", "Fast training", "Works well with linearly separable data"],
                "formula": "P(y=1|x) = 1 / (1 + e^(-z)) where z = β₀ + β₁x₁ + ..."
            },
            "K-Nearest Neighbors": {
                "description": "KNN is called a lazy learning algorithm because it doesn’t build an explicit model during training. Instead, it stores the entire dataset and, when a new data point arrives, looks at the k closest neighbors (based on a distance metric like Euclidean distance) and assigns the class that appears most frequently among them.<br><br>It’s classification by social proof: you are what your nearest neighbors are—assuming you chose a sensible value of k and remembered to scale your features.",
                "when_to_use": "Use when decision boundaries are irregular and you have sufficient training data for local pattern matching.",
                "key_features": ["No training phase", "Simple and intuitive", "Handles multi-class naturally", "Sensitive to feature scaling"],
                "formula": "ŷ = mode(y of k nearest neighbors)"
            },
            "Support Vector Machine": {
                "description": "SVM finds the optimal hyperplane that separates classes while maximizing the margin between them, meaning it chooses the boundary that is as far as possible from the nearest data points of each class (the support vectors). When the data isn’t linearly separable, kernel functions project it into higher-dimensional space, allowing the algorithm to learn non-linear decision boundaries.<br><br>Geometrically elegant, stubbornly precise, and very picky about which points matter—SVMs are minimalists with excellent taste.",
                "when_to_use": "Use for high-dimensional data and when you need a clear margin of separation between classes.",
                "key_features": ["Maximum margin classifier", "Kernel trick for non-linearity", "Effective in high dimensions", "Memory efficient"],
                "formula": "Maximize: margin = 2/||w|| subject to yᵢ(w·xᵢ + b) ≥ 1"
            },
            "Naive Bayes": {
                "description": "Naive Bayes is a probabilistic classification algorithm built on Bayes’ theorem, with the “naive” assumption that features are independent given the class label. While that assumption is often false in the real world, the model works remarkably well in practice—especially for text classification, spam detection, and sentiment analysis.<br><br>It’s a reminder from probability theory that even wrong assumptions can lead to very useful conclusions, as long as the math is honest about them.",
                "when_to_use": "Use for text classification, when features are approximately independent, or when you need very fast predictions.",
                "key_features": ["Very fast training", "Works well with high dimensions", "Handles missing data", "Good baseline model"],
                "formula": "P(y|x) = P(x|y)P(y) / P(x)"
            },
            "Perceptron": {
                "description": "The perceptron is the simplest form of a neural network: a single-layer, binary classifier that computes a weighted sum of inputs and applies a step (threshold) activation function. It learns a linear decision boundary, adjusting weights based on classification errors.<br><br>Historically, it’s where neural networks began—and philosophically, it’s a neuron stripped down to its bare logic: add, compare, decide.",
                "when_to_use": "Use as a simple baseline for linearly separable data or as a building block for understanding neural networks.",
                "key_features": ["Foundation of neural networks", "Online learning capable", "Fast training", "Only for linearly separable data"],
                "formula": "ŷ = sign(w·x + b)"
            }
        }
    },
    "🌳 Tree-Based": {
        "overview": "Tree-based algorithms use a hierarchical, tree-like structure of decisions to make predictions. They recursively split the dataset based on feature values to reduce impurity (for classification) or variance (for regression), forming a sequence of if–then rules. The result is a model that’s often highly interpretable, since you can literally trace the path of a decision from root to leaf.<br><br>They’re intuitive, flexible, and a little dangerous if left unpruned—because trees, like ideas, tend to overgrow when given too much freedom.",
        "use_cases": ["Credit risk assessment", "Customer segmentation", "Feature importance analysis", "Anomaly detection", "Medical decision support"],
        "types": {
            "Decision Tree": {
                "description": "A decision tree is a flowchart-like model where each internal node tests a feature, each branch corresponds to the outcome of that test, and each leaf node produces a final prediction (a class label or a numerical value). The model learns by choosing splits that best separate the data according to measures like Gini impurity, entropy, or variance reduction.<br><br>It’s machine learning’s version of structured common sense: a chain of simple questions that, taken together, lead to a decision.",
                "when_to_use": "Use when you need interpretable models, have mixed feature types, or want to visualize decision rules.",
                "key_features": ["Highly interpretable", "No feature scaling needed", "Handles non-linear relationships", "Prone to overfitting"],
                "formula": "Split criterion: Gini impurity or Information gain"
            },
            "Random Forest": {
                "description": "A random forest is an ensemble method that builds many decision trees, each trained on a random subset of the data (via bootstrapping) and a random subset of features at each split. For classification, the trees vote; for regression, their predictions are averaged.<br><br>The magic is in the disorder. Individual trees are noisy and overconfident, but when you average their opinions, the forest becomes calm, accurate, and far less prone to overfitting.",
                "when_to_use": "Use when you need reliable predictions without much tuning, and can sacrifice some interpretability for accuracy.",
                "key_features": ["Reduces overfitting", "Handles high dimensions", "Provides feature importance", "Robust to outliers"],
                "formula": "ŷ = mode/mean(predictions from n trees)"
            },
            "Gradient Boosting": {
                "description": "Gradient Boosting (specifically Gradient Boosted Decision Trees).<br><br>In gradient boosting, trees are built sequentially, not independently. Each new tree is trained to correct the errors (the residuals) made by the current ensemble, using gradient descent to minimize a specified loss function. Over time, the model focuses more on the hard-to-predict cases.<br><br>If random forests are a wise crowd, gradient boosting is a relentless tutor—each lesson targeted at the mistakes you made last time.",
                "when_to_use": "Use when you need state-of-the-art performance and have time for hyperparameter tuning.",
                "key_features": ["Often best accuracy", "Handles various data types", "Feature importance available", "Risk of overfitting"],
                "formula": "F_m(x) = F_{m-1}(x) + γ·h_m(x)"
            }
        }
    },
    "🔮 Unsupervised": {
        "overview": "Unsupervised learning algorithms work with unlabeled data, aiming to uncover hidden structure rather than predict a known target. They discover natural groupings (clustering), compress or reorganize information (dimensionality reduction), and spot unusual behavior (anomaly or outlier detection). Instead of being told what to look for, these models ask, “What patterns are already here?”<br><br>It’s learning by exploration rather than instruction—less like taking an exam, more like mapping a cave with no signposts.",
        "use_cases": ["Customer segmentation", "Anomaly detection", "Document clustering", "Data compression", "Market basket analysis"],
        "types": {
            "K-Means Clustering": {
                "description": "K-Means is an unsupervised clustering algorithm that partitions data into k clusters. It works iteratively: each data point is assigned to the nearest centroid, then the centroids are updated as the mean of the points in each cluster. This repeats until the assignments stop changing or converge.<br><br>Simple, fast, and surprisingly effective—though it assumes clusters are roughly spherical and makes you choose k up front, which is where the philosophical arguments usually begin.",
                "when_to_use": "Use when you know the number of clusters and need fast, scalable clustering for spherical cluster shapes.",
                "key_features": ["Fast and scalable", "Simple to implement", "Requires k specification", "Assumes spherical clusters"],
                "formula": "Minimize: Σᵢ Σⱼ ||xⱼ - μᵢ||²"
            },
            "DBSCAN": {
                "description": "DBSCAN (Density-Based Spatial Clustering of Applications with Noise).<br><br>DBSCAN groups data points that lie in high-density regions and labels points in low-density areas as outliers or noise. Instead of requiring a fixed number of clusters, it uses two key ideas: a neighborhood radius (ε) and a minimum number of points to define dense regions.<br><br>Its strength is that it can find arbitrarily shaped clusters and handle noise naturally. Its weakness is philosophical and practical: density is a slippery concept, and choosing ε is where human judgment sneaks back into the algorithm.",
                "when_to_use": "Use when clusters have irregular shapes, you don't know the number of clusters, or need to identify outliers.",
                "key_features": ["No k required", "Finds arbitrary shapes", "Identifies outliers", "Sensitive to parameters"],
                "formula": "Core points: |Nₑ(p)| ≥ MinPts"
            },
            "Hierarchical Clustering": {
                "description": "Hierarchical clustering builds a tree-like structure (dendrogram) of clusters by either merging smaller clusters step by step (agglomerative) or splitting a large cluster into smaller ones (divisive). The process relies on distance metrics and linkage criteria to decide which clusters are closest or farthest apart.<br><br>The beauty is that you don’t have to choose the number of clusters in advance. You grow the whole family tree first, then decide where to cut it—classification as genealogy rather than decree.",
                "when_to_use": "Use when you want to explore cluster structure at multiple levels or don't know the optimal number of clusters.",
                "key_features": ["Produces dendrogram", "No k required", "Multiple cluster levels", "Computationally expensive"],
                "formula": "Distance: Single, Complete, Average, or Ward linkage"
            },
            "Gaussian Mixture Model": {
                "description": "Gaussian Mixture Model (GMM).<br><br>A GMM is a probabilistic clustering model that assumes the data is generated from a mixture of several Gaussian (normal) distributions, each with its own mean and covariance, but with unknown parameters. These parameters are typically learned using the Expectation–Maximization (EM) algorithm.<br><br>Unlike k-means, which makes hard assignments, GMMs assign probabilities of belonging to each cluster. Instead of saying “this point is in cluster 2,” it says “this point is 70% likely to be in cluster 2,” which is a much more honest way to describe uncertainty.",
                "when_to_use": "Use for soft clustering when clusters may overlap or when you need probability estimates of cluster membership.",
                "key_features": ["Soft clustering", "Models uncertainty", "Flexible cluster shapes", "Requires cluster count"],
                "formula": "P(x) = Σₖ πₖ · N(x|μₖ, Σₖ)"
            },
            "OPTICS": {
                "description": "OPTICS (Ordering Points To Identify the Clustering Structure).<br><br>OPTICS is a density-based clustering algorithm, closely related to DBSCAN. Instead of producing a single fixed clustering, it creates an ordering of data points that captures the underlying density-based structure. From this ordering (often visualized as a reachability plot), clusters at different density levels can be identified.<br><br>Think of OPTICS as DBSCAN with better memory: rather than committing to one density threshold, it maps the terrain first and lets you see where clusters naturally emerge.",
                "when_to_use": "Use when clusters have varying densities or when you want to visualize clustering structure across scales.",
                "key_features": ["Handles varying densities", "Produces reachability plot", "No strict k needed", "More flexible than DBSCAN"],
                "formula": "Reachability distance: max(core-dist(o), dist(o,p))"
            }
        }
    },
    "🤖 Reinforcement Learning": {
        "overview": "Reinforcement Learning algorithms learn optimal behavior through trial-and-error interaction with an environment. An agent observes the current state, takes an action, and receives a reward in response. Over time, the agent learns a policy—a strategy for choosing actions—that maximizes cumulative (long-term) reward, not just immediate gain.<br><br>It’s learning by consequences rather than instruction. No labels, no answers—just feedback from the universe, which is how animals, humans, and game-playing AIs tend to learn when nobody hands them the solution manual.",
        "use_cases": ["Game playing AI", "Robotics control", "Autonomous vehicles", "Trading strategies", "Recommendation systems"],
        "types": {
            "Q-Learning": {
                "description": "Q-Learning is a model-free reinforcement learning algorithm that learns the value of taking a particular action in a given state, stored in a Q-table. It updates these values using the Bellman equation, iteratively refining estimates of future rewards based on experience.<br><br>What makes Q-learning distinctive is that it’s off-policy: it learns the optimal policy regardless of how the agent currently behaves. In spirit, it’s a bookkeeping system for consequences—keep trying things, remember what worked, and let the math nudge you toward better choices over time.",
                "when_to_use": "Use for discrete state and action spaces when you want an interpretable value function.",
                "key_features": ["Model-free", "Off-policy", "Simple to implement", "Limited to discrete spaces"],
                "formula": "Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]"
            },
            "Deep Q-Network (DQN)": {
                "description": "A DQN extends Q-learning by replacing the Q-table with a neural network that approximates the Q-function. This allows the agent to handle high-dimensional or continuous state spaces—like raw images from a game screen—where a table would be impossibly large. Training uses ideas like experience replay and target networks to stabilize learning.<br><br>In short, it’s Q-learning that grew a cortex. The rules are the same, but memory and generalization scale to worlds far too big for spreadsheets.",
                "when_to_use": "Use for complex environments with high-dimensional states like images or continuous features.",
                "key_features": ["Handles high dimensions", "Experience replay", "Target network stability", "Discrete actions only"],
                "formula": "L(θ) = E[(r + γ·max Q(s',a';θ⁻) - Q(s,a;θ))²]"
            },
            "REINFORCE": {
                "description": "REINFORCE is a policy gradient algorithm that directly optimizes the policy, rather than learning a value function first. It uses Monte Carlo sampling of complete episodes to estimate the gradient of expected return and updates the policy parameters accordingly.<br><br>Because it waits until an episode ends, its gradient estimates are unbiased but noisy—high variance is the price paid for conceptual simplicity. It’s reinforcement learning in its purest form: try a policy, see how the story ends, and adjust your instincts based on the outcome.",
                "when_to_use": "Use when you need continuous action spaces or want to learn stochastic policies directly.",
                "key_features": ["Policy gradient method", "Continuous actions", "High variance", "On-policy learning"],
                "formula": "∇J(θ) = E[∇log π(a|s;θ) · G_t]"
            },
            "Multi-Agent RL": {
                "description": "MARL extends reinforcement learning to settings with multiple agents acting in the same environment. Each agent learns from its own interactions while also being affected by the actions of others, leading to cooperation, competition, or a mix of both. The learning problem becomes harder because the environment is no longer stationary—other agents are learning too.<br><br>It’s where RL stops being a solo puzzle and turns into a social science: strategy, negotiation, emergent behavior, and occasionally chaos—all learned, not programmed.",
                "when_to_use": "Use for multi-player games, distributed systems, or scenarios requiring coordinated agent behavior.",
                "key_features": ["Multiple agents", "Cooperation/competition", "Complex dynamics", "Scalability challenges"],
                "formula": "V^π(s) considering joint actions"
            }
        }
    }
}

CATEGORIES = {
    "🏠 Home": {
        "icon": "🏠",
        "color": "#7c3aed",
        "gradient": "linear-gradient(135deg, #7c3aed, #a855f7)",
        "algorithms": {}
    },
    "📖 Glossary": {
        "icon": "📖",
        "color": "#10b981",
        "gradient": "linear-gradient(135deg, #10b981, #34d399)",
        "algorithms": {}
    },
    "⚖️ Model Compare": {
        "icon": "⚖️",
        "color": "#f59e0b",
        "gradient": "linear-gradient(135deg, #f59e0b, #fbbf24)",
        "algorithms": {}
    },
    "📈 Regression": {
        "icon": "📈",
        "color": "#4f46e5",
        "gradient": "linear-gradient(135deg, #4f46e5, #7c3aed)",
        "algorithms": {
            "Linear Regression": LinearRegressionDemo,
            "Polynomial Regression": PolynomialRegressionDemo,
            "Ridge Regression": RidgeRegressionDemo,
            "Lasso Regression": LassoRegressionDemo,
            "Elastic Net": ElasticNetDemo,
            "Support Vector Regression": SVRDemo,
        }
    },
    "🎯 Classification": {
        "icon": "🎯",
        "color": "#ec4899",
        "gradient": "linear-gradient(135deg, #ec4899, #f472b6)",
        "algorithms": {
            "Logistic Regression": LogisticRegressionDemo,
            "K-Nearest Neighbors": KNNDemo,
            "Support Vector Machine": SVMDemo,
            "Naive Bayes": NaiveBayesDemo,
            "Perceptron": PerceptronDemo,
        }
    },
    "🌳 Tree-Based": {
        "icon": "🌳",
        "color": "#f59e0b",
        "gradient": "linear-gradient(135deg, #f59e0b, #fbbf24)",
        "algorithms": {
            "Decision Tree": DecisionTreeDemo,
            "Random Forest": RandomForestDemo,
            "Gradient Boosting": GradientBoostingDemo,
        }
    },
    "🔮 Unsupervised": {
        "icon": "🔮",
        "color": "#0d9488",
        "gradient": "linear-gradient(135deg, #0d9488, #14b8a6)",
        "algorithms": {
            "K-Means Clustering": KMeansDemo,
            "DBSCAN": DBSCANDemo,
            "Hierarchical Clustering": HierarchicalDemo,
            "Gaussian Mixture Model": GMMDemo,
            "OPTICS": OPTICSDemo,
        }
    },
    "🤖 Reinforcement Learning": {
        "icon": "🤖",
        "color": "#3b82f6",
        "gradient": "linear-gradient(135deg, #3b82f6, #6366f1)",
        "algorithms": {
            "Q-Learning": QLearningDemo,
            "Deep Q-Network (DQN)": DQNDemo,
            "REINFORCE": REINFORCEDemo,
            "Multi-Agent RL": MARLDemo,
        }
    }
}
def render_sidebar():
    """Render the sidebar navigation with enhanced glass effects for light theme"""
    with st.sidebar:
        # Handle navigation from Explore buttons BEFORE rendering widgets
        if "navigate_to_category" in st.session_state and "navigate_to_algo" in st.session_state:
            nav_cat = st.session_state.pop("navigate_to_category")
            nav_algo = st.session_state.pop("navigate_to_algo")
            # Set the widget values directly in session state
            st.session_state["nav_category"] = nav_cat
            st.session_state["algo_select"] = nav_algo
        
        # Enhanced Logo with glow - Clickable to go Home
        st.markdown("""
        <div class="sidebar-logo float-animation">
            <span>🧠</span>
            <span>ML Playground</span>
        </div>
        """, unsafe_allow_html=True)
    
        # Search Box
        st.markdown("---")
        search_query = st.text_input(
            "🔍 Search algorithms...",
            key="sidebar_search",
            placeholder="e.g., regression, SVM..."
        )
        
        # Show search results if query exists
        if search_query:
            results = render_search_box(ALGO_DETAILS, CATEGORIES)
            if results:
                st.markdown("**Search Results:**")
                for r in results[:5]:  # Limit to 5 results
                    if st.button(f"{r['icon']} {r['name']}", key=f"search_{r['name']}", use_container_width=True):
                        st.session_state["navigate_to_category"] = r['category']
                        st.session_state["navigate_to_algo"] = r['name']
                        st.rerun()
            else:
                st.info("No algorithms found")
        
        # Navigation label with glass effect
        st.markdown("---")
        # Category selection with enhanced radio buttons
        category_names = list(CATEGORIES.keys())
        
        selected_category = st.radio(
            "Navigation",
            category_names,
            label_visibility="collapsed",
            key="nav_category"
        )
        
        # Enhanced Algorithm selector
        selected_algo = None
        if selected_category and selected_category not in ["🏠 Home", "📖 Glossary", "⚖️ Model Compare"]:
            cat_data = CATEGORIES[selected_category]
            if cat_data["algorithms"]:
                st.markdown("---")
                st.markdown("""
                <div style="
                    color: #64748b; 
                    font-size: 0.85rem; 
                    margin-bottom: 0.5rem; 
                    padding: 0 0.5rem;
                    font-weight: 600;
                ">
                    🎯 SELECT ALGORITHM
                </div>
                """, unsafe_allow_html=True)
                
                algo_list = ["-- Select --"] + list(cat_data["algorithms"].keys())
                
                selected_algo = st.selectbox(
                    "Algorithm",
                    algo_list,
                    label_visibility="collapsed",
                    key="algo_select"
                )
                if selected_algo == "-- Select --":
                    selected_algo = None
        
        # Quick Actions Section
        st.markdown("---")
        st.markdown("""
        <div style="color: #64748b; font-size: 0.75rem; font-weight: 700; padding: 0 0.5rem; margin-bottom: 0.5rem;">
            ⚡ QUICK ACTIONS
        </div>
        """, unsafe_allow_html=True)
        
        # Surprise Me button
        if st.button("🎲 Surprise Me!", key="surprise_btn", use_container_width=True):
            random_algo = get_random_algorithm(CATEGORIES)
            if random_algo:
                st.session_state["navigate_to_category"] = random_algo["category"]
                st.session_state["navigate_to_algo"] = random_algo["algorithm"]
                st.rerun()
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        # Enhanced Footer
        st.markdown("""
        <div class="sidebar-footer">
            <div class="footer-text">Built with ❤️</div>
            <div class="footer-author">By: CHANDAN RAJ</div>
        </div>
        <br>
        <p style="color: #64748b; font-size: 0.85rem; text-align: center;">
            <a href="https://github.com/chandanraj-03/ML-Learning-Lab" target="_blank" style="color: #6366f1;">source code 🖥️</a>
        </p>
        """, unsafe_allow_html=True)
        
        return selected_category, selected_algo
def render_home():
    """Render the home page with a clean, user-friendly layout"""
    
    # Compact Hero Section
    st.markdown("""
    <div class="hero-compact">
        <div class="hero-content">
            <h1 class="hero-main-title">Welcome to ML Learning Lab</h1>
            <p class="hero-tagline">
                Interactive tutorials for 23+ machine learning algorithms. 
                Learn theory, experiment with live demos, and download production-ready code.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Start Section - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="quick-start-card">
            <div class="qs-header">
                <span class="qs-icon">🚀</span>
                <span class="qs-title">Get Started</span>
            </div>
            <div class="qs-steps">
                <div class="qs-step">
                    <span class="qs-num">1</span>
                    <span class="qs-text">Pick a category from the sidebar</span>
                </div>
                <div class="qs-step">
                    <span class="qs-num">2</span>
                    <span class="qs-text">Select an algorithm to explore</span>
                </div>
                <div class="qs-step">
                    <span class="qs-num">3</span>
                    <span class="qs-text">Learn, experiment, download code</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="quick-start-card">
            <div class="qs-header">
                <span class="qs-icon">✨</span>
                <span class="qs-title">What's Included</span>
            </div>
            <div class="feature-list">
                <div class="feature-item"><span class="feature-check">✓</span> Interactive visualizations</div>
                <div class="feature-item"><span class="feature-check">✓</span> Adjustable parameters</div>
                <div class="feature-item"><span class="feature-check">✓</span> Real-time results</div>
                <div class="feature-item"><span class="feature-check">✓</span> Python code downloads</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Category Cards - Main content
    st.markdown("""
    <div class="section-title-simple">Browse by Category</div>
    """, unsafe_allow_html=True)
    
    def render_category_card(cat_name, cat_data, key_prefix):
        """Helper to render a single category card"""
        icon = cat_data["icon"]
        clean_name = cat_name.split(' ', 1)[1] if ' ' in cat_name else cat_name
        algo_count = len(cat_data["algorithms"])
        color = cat_data["color"]
        
        algos = list(cat_data["algorithms"].keys())[:3]
        algo_tags = "".join([f'<span class="cat-algo-tag">{a}</span>' for a in algos])
        more_text = f"+{algo_count - 3} more" if algo_count > 3 else ""
        
        st.markdown(f"""
        <div class="category-card" style="border-top: 4px solid {color};">
            <div class="cat-header">
                <span class="cat-icon">{icon}</span>
                <div class="cat-info">
                    <span class="cat-name">{clean_name}</span>
                    <span class="cat-count">{algo_count} algorithms</span>
                </div>
            </div>
            <div class="cat-algos">
                {algo_tags}
                <span class="cat-more">{more_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Explore {clean_name}", key=f"{key_prefix}_{clean_name}", use_container_width=True):
            st.session_state["navigate_to_category"] = cat_name
            st.session_state["navigate_to_algo"] = "-- Select --"
            st.rerun()
    
    utility_pages = ["🏠 Home", "📖 Glossary", "⚖️ Model Compare"]
    categories_to_show = [(k, v) for k, v in CATEGORIES.items() if k not in utility_pages]
    
    # Render category cards in rows of 3
    for row_idx in range(0, len(categories_to_show), 3):
        row_cats = categories_to_show[row_idx:row_idx + 3]
        cols = st.columns(3)
        for i, (cat_name, cat_data) in enumerate(row_cats):
            with cols[i]:
                render_category_card(cat_name, cat_data, f"home_cat_r{row_idx}")
        if row_idx + 3 < len(categories_to_show):
            st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Sample Datasets Section
    st.markdown("---")
    from utils.dataset_explorer import render_sample_datasets_section
    render_sample_datasets_section()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Simple footer info
    st.markdown("""
    <div class="home-footer">
        <div class="footer-tip">
            <span class="tip-icon">💡</span>
            <span class="tip-text">Tip: Each algorithm page has Learn, Demo, and Code tabs for a complete learning experience</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
def render_category_home(category_name):
    """Render the category home page with detailed algorithm explanations"""
    cat_data = CATEGORIES[category_name]
    icon = cat_data["icon"]
    clean_name = category_name.split(' ', 1)[1] if ' ' in category_name else category_name
    color = cat_data["color"]
    algo_count = len(cat_data["algorithms"])
    
    # Get category explanation
    cat_explanation = CATEGORY_EXPLANATIONS.get(category_name, {})
    
    # Category header with enhanced styling
    st.markdown(f"""
    <div class="category-overview-header" style="border-left: 5px solid {color};">
        <div class="coh-icon-wrapper" style="background: {color}15;">
            <span class="coh-icon">{icon}</span>
        </div>
        <div class="coh-info">
            <h1 class="coh-title">{clean_name}</h1>
            <p class="coh-subtitle">{algo_count} algorithms to explore</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Category Overview Section
    if cat_explanation.get("overview"):
        st.markdown(f"""
        <div class="category-overview-section">
            <h2 class="cos-title">📖 What is {clean_name}?</h2>
            <p class="cos-text">{cat_explanation["overview"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Use Cases Section
    if cat_explanation.get("use_cases"):
        use_cases_html = "".join([f'<span class="use-case-tag">{uc}</span>' for uc in cat_explanation["use_cases"]])
        st.markdown(f"""
        <div class="use-cases-section">
            <h3 class="ucs-title">🎯 Common Use Cases</h3>
            <div class="use-cases-grid">
                {use_cases_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Algorithm Types Section
    st.markdown(f"""
    <div class="algo-types-header">
        <h2 class="ath-title">🔬 Algorithm Types</h2>
        <p class="ath-subtitle">Click on any algorithm to explore its interactive demo and implementation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display algorithms with detailed explanations
    if cat_data["algorithms"]:
        algo_types = cat_explanation.get("types", {})
        
        for algo_name, algo_class in cat_data["algorithms"].items():
            details = ALGO_DETAILS.get(algo_name, {"desc": "Explore this algorithm", "difficulty": "Intermediate", "icon": "📘"})
            algo_info = algo_types.get(algo_name, {})
            
            diff_colors = {
                "Beginner": "#10b981",
                "Intermediate": "#f59e0b", 
                "Advanced": "#ef4444",
                "Expert": "#7c3aed",
            }
            diff_color = diff_colors.get(details["difficulty"], "#64748b")
            
            # Build key features HTML
            key_features_html = ""
            if algo_info.get("key_features"):
                features = "".join([f'<li class="kf-item">✓ {f}</li>' for f in algo_info["key_features"]])
                key_features_html = f'<ul class="key-features-list">{features}</ul>'
            
            # Build features section
            features_section = ""
            if key_features_html:
                features_section = f'<div class="adc-features"><span class="features-label">Key Features:</span>{key_features_html}</div>'
            
            # Build formula section
            formula_section = ""
            if algo_info.get("formula"):
                formula_section = f'<div class="adc-formula"><span class="formula-label">📐 Formula:</span><code class="formula-code">{algo_info.get("formula", "")}</code></div>'
            
            # Algorithm card with detailed info
            algo_card_html = f"""<div class="algo-detail-card" style="border-left: 4px solid {color};">
<div class="adc-header">
<div class="adc-title-row">
<span class="adc-icon">{details["icon"]}</span>
<div class="adc-title-info">
<span class="adc-name">{algo_name}</span>
<span class="adc-tagline">{details["desc"]}</span>
</div>
<span class="adc-badge" style="background: {diff_color}20; color: {diff_color};">{details["difficulty"]}</span>
</div>
</div>
<div class="adc-body">
<p class="adc-description">{algo_info.get("description", details["desc"])}</p>
<div class="adc-when-to-use">
<span class="wtu-label">💡 When to use:</span>
<span class="wtu-text">{algo_info.get("when_to_use", "Use this algorithm for learning and experimentation.")}</span>
</div>
{features_section}
{formula_section}
</div>
</div>"""
            st.markdown(algo_card_html, unsafe_allow_html=True)
            
            # Start Learning button
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button(f"▶ Try {algo_name.split()[0]}", key=f"btn_{algo_name}", use_container_width=True):
                    st.session_state["navigate_to_category"] = category_name
                    st.session_state["navigate_to_algo"] = algo_name
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("🚧 Algorithms for this category are coming soon!")
    
    # Footer tip
    st.markdown("""
    <div class="category-footer-tip">
        <span class="cft-icon">💡</span>
        <span class="cft-text">Each algorithm page includes an interactive demo where you can adjust parameters and see results in real-time!</span>
    </div>
    """, unsafe_allow_html=True)
@st.dialog("About ML Learning Lab", width="large")
def show_help_dialog():
    """Display the help/about dialog with project details"""
    st.markdown("""
    <style>
    .help-section {
        background: rgba(79, 70, 229, 0.05);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4f46e5;
    }
    .help-section h3 {
        color: #4f46e5;
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
    }
    .help-section p, .help-section li {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .help-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .tech-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #059669;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <span style="font-size: 3rem;">🧠</span>
        <h2 style="margin: 0.5rem 0; color: #1e293b;">ML Learning Lab</h2>
        <p style="color: #64748b; font-size: 0.95rem;">Interactive Machine Learning Education Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview Section
    st.markdown("""
    <div class="help-section">
        <h3>📖 Overview</h3>
        <p>
            <strong>ML Learning Lab</strong> is an interactive, hands-on learning platform built to turn machine learning 
            from abstract math into something you can <strong>see, tweak, and truly understand</strong>. 
            Instead of just reading theory, you experiment with models, observe their behavior in real time, 
            and connect intuition with equations.
        </p>
        <p style="margin-top: 0.75rem;">
            The platform brings together <strong>more than 23 machine learning algorithms</strong> organized 
            across <strong>5 major categories</strong>, guiding you from foundational ideas to practical, 
            production-ready implementations. Each algorithm is presented with clear explanations, interactive 
            visualizations, and downloadable code you can study, modify, and use in real projects.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div class="help-section">
        <h3>✨ Key Features</h3>
        <ul>
            <li><strong>Hands-on Experimentation</strong> - Train models, adjust parameters, and instantly see how predictions and performance change</li>
            <li><strong>Concept Visualizations</strong> - Intuitive plots and animations that reveal how algorithms think, not just what they output</li>
            <li><strong>Structured Learning Paths</strong> - Algorithms grouped by category to help you build knowledge step by step</li>
            <li><strong>Theory & Practice Balance</strong> - Concise explanations paired with real implementations, without unnecessary complexity</li>
            <li><strong>Production-Ready Code</strong> - Clean, well-structured code you can download and integrate into your own projects</li>
            <li><strong>Beginner to Intermediate Friendly</strong> - Designed to grow with you, from first exposure to confident application</li>
        </ul>
        <br>
        <p style="margin-top: 0.75rem; font-style: italic; color: #6366f1;">
            ML Learning Lab is not just about running models. It's about developing intuition, questioning results, 
            and learning machine learning the way it actually works in the real world.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Algorithm Categories
    st.markdown("""
    <div class="help-section">
        <h3>🎯 Algorithm Categories</h3>
        <p>
            <span class="help-badge">📈 Regression (6)</span>
            <span class="help-badge">🎯 Classification (5)</span>
            <span class="help-badge">🌳 Tree-Based (3)</span>
            <span class="help-badge">🔮 Unsupervised (5)</span>
            <span class="help-badge">🤖 Reinforcement Learning (4)</span>
        </p>
        <br>
        <p style="margin-top: 0.75rem; font-style: italic; color: #6366f1;">
            Each category is designed to build on the previous one, so you can learn at your own pace and in your own way.
            <span class="help-badge"> Total Algorithms: 23</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # How to Use Section
    st.markdown("""
    <div class="help-section">
        <h3>🚀 How to Use</h3>
        <ol>
            <li><strong>Navigate</strong> - Use the sidebar to browse categories</li>
            <li><strong>Select</strong> - Choose an algorithm to explore</li>
            <li><strong>Learn</strong> - Read the theory in the "Learn" tab</li>
            <li><strong>Experiment</strong> - Try the interactive demo with custom parameters</li>
            <li><strong>Predict</strong> - Test with your own data inputs</li>
            <li><strong>Download</strong> - Get the Python code for your projects</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Technologies Section
    st.markdown("""
    <div class="help-section">
        <h3>🛠️ Built With</h3>
        <p>
            <span class="tech-badge">Python</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">Scikit-learn</span>
            <span class="tech-badge">NumPy</span>
            <span class="tech-badge">Pandas</span>
            <span class="tech-badge">Matplotlib</span>
            <span class="tech-badge">Plotly</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
        <p style="color: #64748b; font-size: 0.85rem;">
            Created with ❤️ by <strong>Chandan Raj</strong>
        </p>
        <p style="color: #64748b; font-size: 0.85rem;">
            <a href="https://github.com/chandanraj123/ML-Learning-Lab" target="_blank">Click here to learn more or explore the source code🖥️</a>
        </p>
        <p style="color: #94a3b8; font-size: 0.8rem;">
            ML Learning Lab | For Educational Purposes
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Close", use_container_width=True):
        st.rerun()


def main():
    """Main application entry point"""
    # Initialize dark mode in session state
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    
    # Apply dark mode class if enabled
    if st.session_state.dark_mode:
        st.markdown('<div class="dark-mode"></div>', unsafe_allow_html=True)
        st.markdown("""
        <style>
            .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #1e293b 75%, #0f172a 100%) !important; }
            .stApp, .stApp * { color-scheme: dark; }
        </style>
        """, unsafe_allow_html=True)
    
    # Add theme toggle and help button in the top-right header area
    col1, col2, col3 = st.columns([18, 1, 1])
    # Add help button in the top-right header area
    with col2:
        theme_icon = "☀️" if st.session_state.dark_mode else "🌙"
        theme_help = "Switch to Light Mode" if st.session_state.dark_mode else "Switch to Dark Mode"
        if st.button(theme_icon, key="theme_btn", help=theme_help):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    with col3:
        if st.button("ℹ️", key="help_btn", help="About this project"):
            show_help_dialog()
    
    # Render sidebar and get selections
    selected_category, selected_algo = render_sidebar()
    
    if selected_category == "🏠 Home":
        render_home()
    elif selected_category == "📖 Glossary":
        # Render breadcrumb for glossary
        render_breadcrumb(selected_category, None)
        render_glossary_page()
    elif selected_category == "⚖️ Model Compare":
        # Render breadcrumb for model comparison
        render_breadcrumb(selected_category, None)
        render_model_comparison_page()
    elif selected_algo:
        # Render breadcrumb for algorithm page
        render_breadcrumb(selected_category, selected_algo)
        
        # Render specific algorithm
        algo_class = CATEGORIES[selected_category]["algorithms"].get(selected_algo)
        if algo_class:
            try:
                demo = algo_class()
                demo.render()
            except Exception as e:
                st.error(f"Error loading algorithm: {e}")
                st.exception(e)
    else:
        # Render breadcrumb for category page
        render_breadcrumb(selected_category, None)
        # Show category home
        render_category_home(selected_category)

if __name__ == "__main__":
    main()

