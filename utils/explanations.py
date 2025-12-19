"""
Algorithm explanations and theory for ML Portfolio
"""

ALGORITHM_EXPLANATIONS = {
    # Regression Algorithms
    "linear_regression": {
        "name": "Linear Regression",
        "category": "Supervised Learning - Regression",
        "project": "House Price Prediction",
        "description": """
Linear Regression is a fundamental supervised learning algorithm that models the relationship 
between a dependent variable and one or more independent variables by fitting a linear equation.
        """,
        "theory": """
### Mathematical Foundation

The hypothesis function for linear regression is:

$$h_\\theta(x) = \\theta_0 + \\theta_1 x_1 + \\theta_2 x_2 + ... + \\theta_n x_n = \\theta^T x$$

**Objective:** Minimize the Mean Squared Error (MSE):

$$J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2$$

The optimal parameters are found using:
- **Normal Equation:** $\\theta = (X^TX)^{-1}X^Ty$
- **Gradient Descent:** $\\theta_j := \\theta_j - \\alpha \\frac{\\partial}{\\partial \\theta_j} J(\\theta)$
        """,
        "pros": [
            "Simple and interpretable",
            "Fast training and prediction",
            "Works well with linearly separable data",
            "No hyperparameters to tune"
        ],
        "cons": [
            "Assumes linear relationship",
            "Sensitive to outliers",
            "Cannot capture complex patterns",
            "Prone to multicollinearity"
        ],
        "use_cases": [
            "House price prediction",
            "Sales forecasting",
            "Risk assessment",
            "Demand prediction"
        ]
    },
    
    "lasso_regression": {
        "name": "Lasso Regression (L1)",
        "category": "Supervised Learning - Regression",
        "project": "Marketing Channel Analysis",
        "description": """
Lasso (Least Absolute Shrinkage and Selection Operator) regression adds L1 regularization 
to linear regression, which can shrink some coefficients to exactly zero, performing feature selection.
        """,
        "theory": """
### Mathematical Foundation

Lasso minimizes:

$$J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2 + \\lambda \\sum_{j=1}^{n} |\\theta_j|$$

**Key Properties:**
- L1 penalty: $\\lambda ||\\theta||_1$
- Creates sparse solutions (some $\\theta_j = 0$)
- Built-in feature selection
- $\\lambda$ controls regularization strength
        """,
        "pros": [
            "Automatic feature selection",
            "Reduces overfitting",
            "Produces sparse models",
            "Handles multicollinearity"
        ],
        "cons": [
            "May exclude important correlated features",
            "Requires cross-validation for λ",
            "Not suitable for grouped feature selection",
            "Less stable than Ridge"
        ],
        "use_cases": [
            "Feature selection",
            "High-dimensional data",
            "Marketing attribution",
            "Gene expression analysis"
        ]
    },
    
    "ridge_regression": {
        "name": "Ridge Regression (L2)",
        "category": "Supervised Learning - Regression",
        "project": "Student Performance Prediction",
        "description": """
Ridge regression adds L2 regularization to prevent overfitting by penalizing large coefficients, 
shrinking them towards zero but never exactly to zero.
        """,
        "theory": """
### Mathematical Foundation

Ridge minimizes:

$$J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2 + \\lambda \\sum_{j=1}^{n} \\theta_j^2$$

**Closed-form solution:**
$$\\theta = (X^TX + \\lambda I)^{-1}X^Ty$$

**Properties:**
- L2 penalty shrinks but doesn't zero coefficients
- Handles multicollinearity well
- More stable than Lasso
        """,
        "pros": [
            "Handles multicollinearity",
            "More stable than Lasso",
            "Reduces overfitting",
            "Works when features > samples"
        ],
        "cons": [
            "No feature selection",
            "Requires λ tuning",
            "All features retained",
            "Less interpretable than Lasso"
        ],
        "use_cases": [
            "Multicollinear data",
            "Student performance prediction",
            "Economic indicators",
            "Climate modeling"
        ]
    },
    
    "elastic_net": {
        "name": "Elastic Net",
        "category": "Supervised Learning - Regression",
        "project": "Car Price Prediction",
        "description": """
Elastic Net combines L1 and L2 regularization, getting the benefits of both Lasso's 
feature selection and Ridge's coefficient shrinkage.
        """,
        "theory": """
### Mathematical Foundation

Elastic Net minimizes:

$$J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2 + \\lambda_1 \\sum_{j=1}^{n} |\\theta_j| + \\lambda_2 \\sum_{j=1}^{n} \\theta_j^2$$

Or equivalently with mixing parameter $\\rho$:

$$J(\\theta) = \\text{MSE} + \\lambda \\left( \\rho ||\\theta||_1 + \\frac{1-\\rho}{2} ||\\theta||_2^2 \\right)$$

- $\\rho = 1$: Pure Lasso
- $\\rho = 0$: Pure Ridge
        """,
        "pros": [
            "Combines L1 and L2 benefits",
            "Handles correlated features better than Lasso",
            "Feature selection capability",
            "More stable than pure Lasso"
        ],
        "cons": [
            "Two hyperparameters to tune",
            "More complex than Ridge/Lasso",
            "Computationally more expensive",
            "Requires more tuning"
        ],
        "use_cases": [
            "Car price prediction",
            "Genomics",
            "Financial modeling",
            "Text classification"
        ]
    },
    
    "svr": {
        "name": "Support Vector Regression",
        "category": "Supervised Learning - Regression",
        "project": "Stock Price Forecasting",
        "description": """
SVR applies Support Vector Machine concepts to regression, using kernel tricks 
to model nonlinear relationships while maintaining regularization.
        """,
        "theory": """
### Mathematical Foundation

SVR finds a function $f(x)$ that deviates from $y$ by at most $\\epsilon$:

$$\\min \\frac{1}{2}||w||^2 + C \\sum_{i=1}^{m}(\\xi_i + \\xi_i^*)$$

Subject to:
- $y_i - f(x_i) \\leq \\epsilon + \\xi_i$
- $f(x_i) - y_i \\leq \\epsilon + \\xi_i^*$

**Kernels:**
- Linear: $K(x,x') = x^T x'$
- RBF: $K(x,x') = \\exp(-\\gamma ||x-x'||^2)$
- Polynomial: $K(x,x') = (\\gamma x^T x' + r)^d$
        """,
        "pros": [
            "Handles nonlinear relationships",
            "Robust to outliers (ε-insensitive)",
            "Works well in high dimensions",
            "Kernel flexibility"
        ],
        "cons": [
            "Slow on large datasets",
            "Memory intensive",
            "Requires feature scaling",
            "Hyperparameter sensitive"
        ],
        "use_cases": [
            "Stock price forecasting",
            "Time series prediction",
            "Energy consumption",
            "Weather forecasting"
        ]
    },
    
    "polynomial_regression": {
        "name": "Polynomial Regression",
        "category": "Supervised Learning - Regression",
        "project": "Battery Degradation Prediction",
        "description": """
Polynomial regression extends linear regression by adding polynomial features, 
allowing it to model nonlinear relationships while still using linear methods.
        """,
        "theory": """
### Mathematical Foundation

Transform features with polynomial degree $d$:

$$h_\\theta(x) = \\theta_0 + \\theta_1 x + \\theta_2 x^2 + ... + \\theta_d x^d$$

**For multiple features:**
$$\\phi(x) = [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2, ...]$$

**Key Considerations:**
- Higher degree = more flexibility = risk of overfitting
- Use cross-validation to select optimal degree
- Often combined with regularization
        """,
        "pros": [
            "Captures nonlinear patterns",
            "Simple extension of linear regression",
            "Interpretable coefficients",
            "Works with standard linear methods"
        ],
        "cons": [
            "Prone to overfitting with high degree",
            "Feature explosion with many variables",
            "Sensitive to extrapolation",
            "Requires degree selection"
        ],
        "use_cases": [
            "Battery degradation curves",
            "Growth modeling",
            "Physics relationships",
            "Signal processing"
        ]
    },
    
    # Classification Algorithms
    "logistic_regression": {
        "name": "Logistic Regression",
        "category": "Supervised Learning - Classification",
        "project": "Email Spam Detection",
        "description": """
Logistic Regression is a classification algorithm that uses the logistic (sigmoid) function 
to model the probability of a binary outcome.
        """,
        "theory": """
### Mathematical Foundation

**Sigmoid Function:**
$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$

**Hypothesis:**
$$h_\\theta(x) = \\sigma(\\theta^T x) = P(y=1|x;\\theta)$$

**Cross-Entropy Loss:**
$$J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m} [y^{(i)} \\log(h_\\theta(x^{(i)})) + (1-y^{(i)}) \\log(1-h_\\theta(x^{(i)}))]$$

**Decision Boundary:** Predict 1 if $h_\\theta(x) \\geq 0.5$
        """,
        "pros": [
            "Probabilistic interpretation",
            "Fast and efficient",
            "Highly interpretable",
            "Works well with linearly separable data"
        ],
        "cons": [
            "Assumes linear decision boundary",
            "Cannot solve nonlinear problems",
            "Sensitive to outliers",
            "Requires feature scaling"
        ],
        "use_cases": [
            "Spam detection",
            "Credit risk assessment",
            "Disease diagnosis",
            "Click prediction"
        ]
    },
    
    "knn": {
        "name": "K-Nearest Neighbors",
        "category": "Supervised Learning - Classification",
        "project": "Movie Recommendation",
        "description": """
KNN is a non-parametric algorithm that classifies a point based on the majority class 
of its k nearest neighbors in the feature space.
        """,
        "theory": """
### Mathematical Foundation

**Distance Metrics:**
- Euclidean: $d(x,y) = \\sqrt{\\sum_{i=1}^{n}(x_i - y_i)^2}$
- Manhattan: $d(x,y) = \\sum_{i=1}^{n}|x_i - y_i|$
- Minkowski: $d(x,y) = (\\sum_{i=1}^{n}|x_i - y_i|^p)^{1/p}$

**Prediction:**
$$\\hat{y} = \\text{mode}(y_i : x_i \\in N_k(x))$$

where $N_k(x)$ is the set of k nearest neighbors.

**Weighted KNN:** Weight votes by inverse distance
        """,
        "pros": [
            "No training phase",
            "Naturally handles multi-class",
            "Non-parametric, no assumptions",
            "Simple to understand"
        ],
        "cons": [
            "Slow prediction for large datasets",
            "Memory intensive",
            "Sensitive to irrelevant features",
            "Curse of dimensionality"
        ],
        "use_cases": [
            "Recommendation systems",
            "Image classification",
            "Pattern recognition",
            "Anomaly detection"
        ]
    },
    
    "svm": {
        "name": "Support Vector Machine",
        "category": "Supervised Learning - Classification",
        "project": "Parkinson's Disease Prediction",
        "description": """
SVM finds the optimal hyperplane that maximizes the margin between classes, 
with kernel tricks enabling nonlinear classification.
        """,
        "theory": """
### Mathematical Foundation

**Objective:** Maximize margin $\\frac{2}{||w||}$

$$\\min_{w,b} \\frac{1}{2}||w||^2 + C \\sum_{i=1}^{m} \\xi_i$$

Subject to: $y^{(i)}(w^T x^{(i)} + b) \\geq 1 - \\xi_i$

**Dual Form with Kernels:**
$$\\max_\\alpha \\sum_i \\alpha_i - \\frac{1}{2} \\sum_{i,j} \\alpha_i \\alpha_j y_i y_j K(x_i, x_j)$$

**Popular Kernels:**
- Linear, RBF, Polynomial, Sigmoid
        """,
        "pros": [
            "Effective in high dimensions",
            "Memory efficient (uses support vectors)",
            "Kernel flexibility",
            "Strong theoretical foundation"
        ],
        "cons": [
            "Slow on large datasets",
            "Sensitive to feature scaling",
            "No probability estimates by default",
            "Hyperparameter tuning required"
        ],
        "use_cases": [
            "Medical diagnosis",
            "Image classification",
            "Text categorization",
            "Bioinformatics"
        ]
    },
    
    "naive_bayes": {
        "name": "Naive Bayes",
        "category": "Supervised Learning - Classification",
        "project": "Sentiment Analysis",
        "description": """
Naive Bayes applies Bayes' theorem with the "naive" assumption that features 
are conditionally independent given the class label.
        """,
        "theory": """
### Mathematical Foundation

**Bayes' Theorem:**
$$P(y|x) = \\frac{P(x|y) P(y)}{P(x)}$$

**Naive Assumption:**
$$P(x|y) = \\prod_{j=1}^{n} P(x_j|y)$$

**Types:**
- **Gaussian NB:** $P(x_j|y) = \\mathcal{N}(\\mu_{jy}, \\sigma^2_{jy})$
- **Multinomial NB:** For text/count data
- **Bernoulli NB:** For binary features

**Decision:** $\\hat{y} = \\arg\\max_y P(y) \\prod_{j=1}^{n} P(x_j|y)$
        """,
        "pros": [
            "Fast training and prediction",
            "Works well with high-dimensional data",
            "Handles missing data",
            "Good for text classification"
        ],
        "cons": [
            "Assumes feature independence",
            "Cannot learn feature interactions",
            "Zero frequency problem",
            "Poor probability estimates"
        ],
        "use_cases": [
            "Sentiment analysis",
            "Spam filtering",
            "Document classification",
            "Medical diagnosis"
        ]
    },
    
    "perceptron": {
        "name": "Perceptron",
        "category": "Supervised Learning - Classification",
        "project": "Handwritten Digit Classification",
        "description": """
The Perceptron is the simplest neural network—a single neuron that learns a linear 
decision boundary through iterative weight updates.
        """,
        "theory": """
### Mathematical Foundation

**Perceptron Model:**
$$\\hat{y} = \\text{sign}(w^T x + b)$$

**Update Rule (if misclassified):**
$$w := w + \\eta y^{(i)} x^{(i)}$$
$$b := b + \\eta y^{(i)}$$

**Convergence:**
- Guaranteed to converge if data is linearly separable
- Number of mistakes bounded by $(R/\\gamma)^2$
  - $R$: max norm of input
  - $\\gamma$: margin of separator
        """,
        "pros": [
            "Simple and fast",
            "Online learning capable",
            "Guaranteed convergence for separable data",
            "Foundation of neural networks"
        ],
        "cons": [
            "Only linear boundaries",
            "No convergence for non-separable data",
            "Sensitive to feature scaling",
            "No probability estimates"
        ],
        "use_cases": [
            "Binary classification",
            "Online learning",
            "Simple pattern recognition",
            "Educational purposes"
        ]
    },
    
    # Tree-Based Models
    "decision_tree": {
        "name": "Decision Tree",
        "category": "Tree-Based Learning",
        "project": "Loan Approval Prediction",
        "description": """
Decision Trees learn hierarchical decision rules by recursively splitting data 
based on feature thresholds that maximize information gain.
        """,
        "theory": """
### Mathematical Foundation

**Splitting Criteria:**

**Gini Impurity:**
$$G(p) = 1 - \\sum_{k=1}^{K} p_k^2$$

**Entropy:**
$$H(p) = -\\sum_{k=1}^{K} p_k \\log_2(p_k)$$

**Information Gain:**
$$IG = H(parent) - \\sum_{c} \\frac{n_c}{n} H(child_c)$$

**Pruning:** Reduce overfitting by limiting depth or minimum samples
        """,
        "pros": [
            "Highly interpretable",
            "No feature scaling needed",
            "Handles mixed data types",
            "Feature importance built-in"
        ],
        "cons": [
            "Prone to overfitting",
            "Unstable (high variance)",
            "Biased toward features with many levels",
            "Cannot extrapolate"
        ],
        "use_cases": [
            "Loan approval",
            "Medical diagnosis",
            "Customer segmentation",
            "Rule extraction"
        ]
    },
    
    "random_forest": {
        "name": "Random Forest",
        "category": "Tree-Based Learning",
        "project": "Customer Churn Prediction",
        "description": """
Random Forest is an ensemble of decision trees trained on random subsets of data 
and features, reducing overfitting through averaging.
        """,
        "theory": """
### Mathematical Foundation

**Bagging + Feature Randomness:**

1. Sample $B$ bootstrap datasets
2. For each tree, at each split:
   - Randomly select $m$ features (typically $m = \\sqrt{p}$)
   - Find best split among those $m$ features
3. Aggregate predictions:
   - Classification: Majority vote
   - Regression: Average

**Out-of-Bag Error:** Natural validation using samples not in bootstrap
        """,
        "pros": [
            "Reduces overfitting vs single tree",
            "Handles high-dimensional data",
            "Robust to outliers",
            "Feature importance ranking"
        ],
        "cons": [
            "Less interpretable than single tree",
            "Memory intensive",
            "Slower training/prediction",
            "Cannot extrapolate"
        ],
        "use_cases": [
            "Customer churn prediction",
            "Credit scoring",
            "Medical diagnosis",
            "Feature selection"
        ]
    },
    
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "category": "Tree-Based Learning",
        "project": "Fraud Detection",
        "description": """
Gradient Boosting builds trees sequentially, with each tree correcting the errors 
of the previous ensemble using gradient descent.
        """,
        "theory": """
### Mathematical Foundation

**Iterative Improvement:**
$$F_m(x) = F_{m-1}(x) + \\gamma_m h_m(x)$$

where $h_m$ is fitted to negative gradients:
$$h_m = \\arg\\min_h \\sum_{i=1}^{n} \\left( -\\frac{\\partial L(y_i, F_{m-1}(x_i))}{\\partial F_{m-1}(x_i)} - h(x_i) \\right)^2$$

**Popular Implementations:**
- **XGBoost:** Regularization + efficient implementation
- **LightGBM:** Leaf-wise growth + histogram binning
- **CatBoost:** Ordered boosting + categorical features
        """,
        "pros": [
            "State-of-the-art accuracy",
            "Handles mixed data types",
            "Built-in regularization",
            "Feature importance"
        ],
        "cons": [
            "Sensitive to hyperparameters",
            "Can overfit without tuning",
            "Sequential training (slower)",
            "Less interpretable"
        ],
        "use_cases": [
            "Fraud detection",
            "Ranking systems",
            "Competition winning models",
            "Click-through prediction"
        ]
    },
    
    # Unsupervised Learning
    "kmeans": {
        "name": "K-Means Clustering",
        "category": "Unsupervised Learning",
        "project": "Customer Segmentation",
        "description": """
K-Means partitions data into k clusters by iteratively assigning points to the 
nearest centroid and updating centroids to cluster means.
        """,
        "theory": """
### Mathematical Foundation

**Objective:** Minimize within-cluster sum of squares:
$$J = \\sum_{j=1}^{k} \\sum_{x_i \\in C_j} ||x_i - \\mu_j||^2$$

**Lloyd's Algorithm:**
1. Initialize k centroids randomly
2. **Assign:** Each point to nearest centroid
3. **Update:** Recompute centroids as cluster means
4. Repeat until convergence

**K-Means++:** Smart initialization for better convergence
        """,
        "pros": [
            "Simple and fast",
            "Scales to large datasets",
            "Easy to interpret",
            "Guaranteed convergence"
        ],
        "cons": [
            "Must specify k beforehand",
            "Sensitive to initialization",
            "Assumes spherical clusters",
            "Affected by outliers"
        ],
        "use_cases": [
            "Customer segmentation",
            "Image compression",
            "Document clustering",
            "Anomaly detection"
        ]
    },
    
    "dbscan": {
        "name": "DBSCAN",
        "category": "Unsupervised Learning",
        "project": "Network Anomaly Detection",
        "description": """
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds 
clusters of arbitrary shape based on local density.
        """,
        "theory": """
### Mathematical Foundation

**Core Concepts:**
- **ε-neighborhood:** $N_\\epsilon(p) = \\{q : d(p,q) < \\epsilon\\}$
- **Core point:** $|N_\\epsilon(p)| \\geq \\text{minPts}$
- **Border point:** Not core but in core's neighborhood
- **Noise point:** Neither core nor border

**Algorithm:**
1. Find core, border, noise points
2. Connect core points within ε of each other
3. Assign border points to nearest core's cluster
4. Mark noise as outliers
        """,
        "pros": [
            "No need to specify k",
            "Finds arbitrary cluster shapes",
            "Identifies outliers naturally",
            "Robust to noise"
        ],
        "cons": [
            "Struggles with varying densities",
            "Sensitive to ε and minPts",
            "Not good for high dimensions",
            "Memory intensive"
        ],
        "use_cases": [
            "Anomaly detection",
            "Spatial data analysis",
            "Image segmentation",
            "Network intrusion detection"
        ]
    },
    
    "hierarchical": {
        "name": "Hierarchical Clustering",
        "category": "Unsupervised Learning",
        "project": "Song Grouping",
        "description": """
Agglomerative hierarchical clustering builds a tree of clusters by iteratively 
merging the closest clusters based on a linkage criterion.
        """,
        "theory": """
### Mathematical Foundation

**Linkage Methods:**

- **Single:** $d(A,B) = \\min_{a \\in A, b \\in B} d(a,b)$
- **Complete:** $d(A,B) = \\max_{a \\in A, b \\in B} d(a,b)$
- **Average:** $d(A,B) = \\frac{1}{|A||B|} \\sum_{a,b} d(a,b)$
- **Ward:** Minimize variance increase

**Algorithm:**
1. Start with each point as a cluster
2. Find closest pair of clusters
3. Merge them
4. Repeat until one cluster or k clusters
        """,
        "pros": [
            "Dendrogram visualization",
            "No need to specify k upfront",
            "Deterministic results",
            "Works with any distance metric"
        ],
        "cons": [
            "O(n²) space and time",
            "Cannot undo merges",
            "Sensitive to noise",
            "Different linkages give different results"
        ],
        "use_cases": [
            "Music/song grouping",
            "Taxonomy creation",
            "Gene expression analysis",
            "Social network analysis"
        ]
    },
    
    "gmm": {
        "name": "Gaussian Mixture Models",
        "category": "Unsupervised Learning",
        "project": "Image Segmentation",
        "description": """
GMM assumes data is generated from a mixture of Gaussian distributions 
and uses EM algorithm to estimate the parameters.
        """,
        "theory": """
### Mathematical Foundation

**Model:**
$$p(x) = \\sum_{k=1}^{K} \\pi_k \\mathcal{N}(x | \\mu_k, \\Sigma_k)$$

**EM Algorithm:**

**E-step:** Compute responsibilities
$$\\gamma_{nk} = \\frac{\\pi_k \\mathcal{N}(x_n|\\mu_k, \\Sigma_k)}{\\sum_j \\pi_j \\mathcal{N}(x_n|\\mu_j, \\Sigma_j)}$$

**M-step:** Update parameters
$$\\mu_k = \\frac{\\sum_n \\gamma_{nk} x_n}{\\sum_n \\gamma_{nk}}$$
        """,
        "pros": [
            "Soft clustering (probabilities)",
            "Handles elliptical clusters",
            "Flexible covariance options",
            "Density estimation"
        ],
        "cons": [
            "Requires specifying k",
            "Sensitive to initialization",
            "Can converge to local optima",
            "Assumes Gaussian components"
        ],
        "use_cases": [
            "Image segmentation",
            "Color quantization",
            "Speaker recognition",
            "Density estimation"
        ]
    },
    
    "optics": {
        "name": "OPTICS",
        "category": "Unsupervised Learning",
        "project": "IoT Sensor Outlier Detection",
        "description": """
OPTICS (Ordering Points To Identify the Clustering Structure) creates an ordering 
that captures density-based clustering structure at all scales.
        """,
        "theory": """
### Mathematical Foundation

**Key Concepts:**
- **Core distance:** Minimum ε to make point a core point
$$\\text{core-dist}(p) = \\begin{cases} \\text{undefined} & |N_\\epsilon(p)| < \\text{minPts} \\\\ d(p, N^{\\text{minPts}}) & \\text{otherwise} \\end{cases}$$

- **Reachability distance:** 
$$\\text{reach-dist}(o, p) = \\max(\\text{core-dist}(o), d(o, p))$$

**Algorithm:**
1. Compute reachability distances
2. Order points by reachability
3. Extract clusters using reachability plot
        """,
        "pros": [
            "Works with varying densities",
            "Produces cluster hierarchy",
            "Less sensitive to parameters",
            "Finds nested clusters"
        ],
        "cons": [
            "Computationally expensive",
            "Requires minPts parameter",
            "Complex to interpret",
            "Memory intensive"
        ],
        "use_cases": [
            "IoT sensor analysis",
            "Multi-density clustering",
            "Outlier detection",
            "Spatial pattern mining"
        ]
    },
    
    # Reinforcement Learning
    "q_learning": {
        "name": "Q-Learning",
        "category": "Reinforcement Learning",
        "project": "Taxi Navigation",
        "description": """
Q-Learning is a model-free RL algorithm that learns the optimal action-value function 
by iteratively updating Q-values based on rewards and future value estimates.
        """,
        "theory": """
### Mathematical Foundation

**Q-Value Update:**
$$Q(s, a) \\leftarrow Q(s, a) + \\alpha [r + \\gamma \\max_{a'} Q(s', a') - Q(s, a)]$$

**Components:**
- $\\alpha$: Learning rate
- $\\gamma$: Discount factor
- $r$: Immediate reward
- $\\max_{a'} Q(s', a')$: Best future value

**Exploration:** ε-greedy policy
$$a = \\begin{cases} \\text{random} & \\text{with prob } \\epsilon \\\\ \\arg\\max_a Q(s,a) & \\text{otherwise} \\end{cases}$$
        """,
        "pros": [
            "Model-free",
            "Guaranteed convergence",
            "Off-policy learning",
            "Simple to implement"
        ],
        "cons": [
            "Tabular (limited to small state spaces)",
            "Slow convergence",
            "Exploration-exploitation tradeoff",
            "Memory intensive for large spaces"
        ],
        "use_cases": [
            "Grid world navigation",
            "Game playing",
            "Robot control",
            "Resource management"
        ]
    },
    
    "dqn": {
        "name": "Deep Q-Network (DQN)",
        "category": "Reinforcement Learning",
        "project": "Atari Game Playing",
        "description": """
DQN combines Q-Learning with deep neural networks to handle high-dimensional 
state spaces, using experience replay and target networks for stability.
        """,
        "theory": """
### Mathematical Foundation

**DQN Loss:**
$$L(\\theta) = \\mathbb{E}[(r + \\gamma \\max_{a'} Q(s', a'; \\theta^-) - Q(s, a; \\theta))^2]$$

**Key Innovations:**
1. **Experience Replay:** Store and sample transitions $(s, a, r, s')$
2. **Target Network:** Separate network $\\theta^-$ updated periodically
3. **Convolutional Layers:** Process raw pixel input

**Double DQN:** Reduce overestimation
$$y = r + \\gamma Q(s', \\arg\\max_{a'} Q(s', a'; \\theta); \\theta^-)$$
        """,
        "pros": [
            "Handles high-dimensional states",
            "End-to-end learning from pixels",
            "Stable training with replay",
            "Human-level performance on games"
        ],
        "cons": [
            "High computational cost",
            "Hyperparameter sensitive",
            "Only discrete actions",
            "May overestimate values"
        ],
        "use_cases": [
            "Atari games",
            "Video game AI",
            "Robotic control",
            "Autonomous systems"
        ]
    },
    
    "reinforce": {
        "name": "REINFORCE",
        "category": "Reinforcement Learning",
        "project": "News Recommendation",
        "description": """
REINFORCE is a policy gradient method that directly learns a parameterized policy 
by estimating gradients from sampled trajectories.
        """,
        "theory": """
### Mathematical Foundation

**Policy Gradient Theorem:**
$$\\nabla_\\theta J(\\theta) = \\mathbb{E}_\\pi [\\nabla_\\theta \\log \\pi_\\theta(a|s) Q^\\pi(s, a)]$$

**REINFORCE Update:**
$$\\theta \\leftarrow \\theta + \\alpha \\sum_{t=0}^{T} \\nabla_\\theta \\log \\pi_\\theta(a_t|s_t) G_t$$

where $G_t = \\sum_{k=t}^{T} \\gamma^{k-t} r_k$ is the return.

**Baseline:** Reduce variance with $G_t - b(s_t)$
        """,
        "pros": [
            "Handles continuous actions",
            "Directly optimizes policy",
            "Can learn stochastic policies",
            "Theoretically grounded"
        ],
        "cons": [
            "High variance",
            "Sample inefficient",
            "Hyperparameter sensitive",
            "Slow convergence"
        ],
        "use_cases": [
            "Recommendation systems",
            "Dialogue systems",
            "Continuous control",
            "NLP tasks"
        ]
    },
    
    "marl": {
        "name": "Multi-Agent RL",
        "category": "Reinforcement Learning",
        "project": "Multi-Robot Warehouse",
        "description": """
Multi-Agent RL extends RL to multiple interacting agents that may cooperate, 
compete, or have mixed objectives in a shared environment.
        """,
        "theory": """
### Mathematical Foundation

**Stochastic Game Framework:**
- $N$ agents with policies $\\pi_1, ..., \\pi_N$
- Joint action: $a = (a_1, ..., a_N)$
- Each agent $i$ maximizes:
$$V_i^\\pi = \\mathbb{E}[\\sum_{t=0}^{\\infty} \\gamma^t r_i(s_t, a_t)]$$

**Approaches:**
- **Independent Q-Learning:** Each agent learns independently
- **Centralized Training, Decentralized Execution (CTDE)**
- **Communication:** Agents share information
- **Value Decomposition:** $Q_{total} = f(Q_1, ..., Q_n)$
        """,
        "pros": [
            "Models real-world multi-agent systems",
            "Emergent coordination",
            "Scalable to many agents",
            "Handles complex interactions"
        ],
        "cons": [
            "Non-stationary environment",
            "Credit assignment problem",
            "Exponential action space",
            "Training instability"
        ],
        "use_cases": [
            "Warehouse robotics",
            "Traffic control",
            "Multi-player games",
            "Autonomous vehicles"
        ]
    }
}


def get_explanation(algorithm_key: str) -> dict:
    """Get explanation for a specific algorithm"""
    return ALGORITHM_EXPLANATIONS.get(algorithm_key, {})


def get_all_algorithms() -> dict:
    """Get all algorithm explanations"""
    return ALGORITHM_EXPLANATIONS


def get_algorithms_by_category(category: str) -> dict:
    """Get algorithms filtered by category"""
    return {k: v for k, v in ALGORITHM_EXPLANATIONS.items() 
            if category.lower() in v.get('category', '').lower()}
