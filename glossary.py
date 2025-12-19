GLOSSARY_TERMS = {

    "Accuracy": {
        "definition": (
            "Accuracy measures the proportion of correct predictions made by a model out of all predictions.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Overall Performance Metric</strong>: Calculates (Correct Predictions) / (Total Predictions)<br>"
            "• <strong>Balanced Dataset Suitability</strong>: Most meaningful when classes are approximately equally distributed<br>"
            "• <strong>Limitation with Imbalance</strong>: Can be misleading (e.g., 95% accuracy with 95% majority class)<br>"
            "• <strong>Error Type Insensitivity</strong>: Doesn't differentiate between false positives and false negatives<br>"
            "• <strong>Binary & Multiclass</strong>: Applicable to both binary and multiclass classification<br>"
            "• <strong>Baseline Metric</strong>: Useful initial assessment but often insufficient alone<br><br>"
            "When to Use:<br>"
            "• Balanced classification problems<br>"
            "• Initial model assessment<br>"
            "• Situations where all error types have equal cost<br><br>"
            "Mathematical Formulation:<br>"
            "Accuracy = (True Positives + True Negatives) / (TP + TN + False Positives + False Negatives)<br><br>"
            "Practical Example:<br>"
            "In a medical test with 100 patients (90 healthy, 10 diseased):<br>"
            "• Model that predicts 'healthy' for everyone achieves 90% accuracy<br>"
            "• This highlights why accuracy alone can be deceptive"
        ),
        "category": "Evaluation Metrics",
        "icon": "🎯",
        "complexity": "Beginner",
        "related_terms": ["Precision", "Recall", "F1 Score", "Confusion Matrix"]
    },

    "Precision": {
        "definition": (
            "Precision (Positive Predictive Value) measures the proportion of true positive predictions among all positive predictions.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Positive Prediction Quality</strong>: Answers 'How reliable are our positive predictions?'<br>"
            "• <strong>False Positive Focus</strong>: Penalizes false alarms (Type I errors)<br>"
            "• <strong>Critical Applications</strong>: Spam detection, fraud prevention, content moderation<br>"
            "• <strong>Trade-off with Recall</strong>: Increasing precision typically decreases recall<br>"
            "• <strong>Class-Specific</strong>: Can be calculated per class in multiclass problems<br>"
            "• <strong>Business Impact</strong>: High precision reduces operational costs from false alarms<br><br>"
            "When to Prioritize:<br>"
            "• When false positives are expensive or harmful<br>"
            "• Customer-facing automated decisions<br>"
            "• Legal or compliance-sensitive applications<br><br>"
            "Mathematical Formulation:<br>"
            "Precision = True Positives / (True Positives + False Positives)<br><br>"
            "Example Scenario - Email Spam Filter:<br>"
            "• High precision: When filter marks email as spam, it's almost certainly spam<br>"
            "• Low precision: Many legitimate emails incorrectly marked as spam"
        ),
        "category": "Evaluation Metrics",
        "icon": "🔬",
        "complexity": "Intermediate",
        "related_terms": ["Recall", "F1 Score", "Specificity", "Confusion Matrix"]
    },

    "Recall (Sensitivity)": {
        "definition": (
            "Recall measures the proportion of actual positives correctly identified by the model.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Completeness Metric</strong>: Answers 'What fraction of actual positives did we catch?'<br>"
            "• <strong>False Negative Focus</strong>: Penalizes missed detections (Type II errors)<br>"
            "• <strong>Critical Applications</strong>: Medical diagnosis, safety systems, search & retrieval<br>"
            "• <strong>Alternative Names</strong>: Sensitivity, True Positive Rate, Hit Rate<br>"
            "• <strong>Recall-Precision Trade-off</strong>: Capturing more positives usually increases false positives<br>"
            "• <strong>Life-Saving Importance</strong>: Often prioritized in healthcare and safety-critical systems<br><br>"
            "When to Prioritize:<br>"
            "• Medical screening (cancer detection)<br>"
            "• Fraud detection where missed fraud is costly<br>"
            "• Search engines (want all relevant documents)<br><br>"
            "Mathematical Formulation:<br>"
            "Recall = True Positives / (True Positives + False Negatives)<br><br>"
            "Example Scenario - Cancer Screening:<br>"
            "• High recall: Very few cancer cases are missed (critical for patient safety)<br>"
            "• Low recall: Many cancer cases go undetected (potentially fatal)"
        ),
        "category": "Evaluation Metrics",
        "icon": "📡",
        "complexity": "Intermediate",
        "related_terms": ["Precision", "F1 Score", "Specificity", "ROC Curve"]
    },

    "F1 Score": {
        "definition": (
            "F1 Score is the harmonic mean of precision and recall, providing a single balanced metric.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Balanced Metric</strong>: Harmonically combines precision and recall<br>"
            "• <strong>Imbalance Robustness</strong>: More informative than accuracy with skewed classes<br>"
            "• <strong>Harmonic Mean Property</strong>: Penalizes extreme values more than arithmetic mean<br>"
            "• <strong>Binary Focus</strong>: Originally for binary classification; multiclass requires averaging<br>"
            "• <strong>Optimization Target</strong>: Often used when seeking balance between precision/recall<br>"
            "• <strong>Limitation</strong>: Assumes equal importance of precision and recall<br><br>"
            "When to Use:<br>"
            "• Class-imbalanced datasets<br>"
            "• When both false positives and false negatives matter<br>"
            "• Model comparison with single metric preference<br><br>"
            "Mathematical Formulation:<br>"
            "F1 = 2 × (Precision × Recall) / (Precision + Recall)<br>"
            "Fβ Score (generalized): (1+β²) × (Precision × Recall) / (β²×Precision + Recall)<br><br>"
            "Interpretation Guidelines:<br>"
            "• F1 > 0.9: Excellent performance<br>"
            "• F1 0.7-0.9: Good performance<br>"
            "• F1 < 0.7: Needs improvement"
        ),
        "category": "Evaluation Metrics",
        "icon": "⚖️",
        "complexity": "Intermediate",
        "related_terms": ["Precision", "Recall", "Fβ Score", "ROC-AUC"]
    },

    "AUC-ROC (Area Under ROC Curve)": {
        "definition": (
            "AUC-ROC measures a classifier's ability to distinguish between classes across all thresholds.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Threshold-Agnostic</strong>: Evaluates performance across all classification thresholds<br>"
            "• <strong>Ranking Quality</strong>: Measures probability that positive samples rank higher than negatives<br>"
            "• <strong>Scale Interpretation</strong>: 0.5 (random) to 1.0 (perfect discrimination)<br>"
            "• <strong>Imbalance Resilience</strong>: Robust to class distribution changes<br>"
            "• <strong>Visual Representation</strong>: ROC curve plots TPR vs FPR at various thresholds<br>"
            "• <strong>Limitation</strong>: Can be optimistic with severe class imbalance<br><br>"
            "When to Use:<br>"
            "• Binary classification model comparison<br>"
            "• Threshold selection analysis<br>"
            "• Medical test evaluation<br><br>"
            "Interpretation Guidelines:<br>"
            "• AUC = 0.5: No discrimination (random)<br>"
            "• AUC 0.7-0.8: Acceptable discrimination<br>"
            "• AUC 0.8-0.9: Excellent discrimination<br>"
            "• AUC > 0.9: Outstanding discrimination<br><br>"
            "ROC Curve Components:<br>"
            "• X-axis: False Positive Rate (1 - Specificity)<br>"
            "• Y-axis: True Positive Rate (Recall/Sensitivity)<br>"
            "• Diagonal: Random classifier baseline"
        ),
        "category": "Evaluation Metrics",
        "icon": "📈",
        "complexity": "Advanced",
        "related_terms": ["ROC Curve", "Precision-Recall Curve", "Specificity", "Youden's Index"]
    },

    "Overfitting": {
        "definition": (
            "Overfitting occurs when a model learns patterns specific to the training data that don't generalize.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>High Training Performance</strong>: Exceptionally low error on training data<br>"
            "• <strong>Poor Generalization</strong>: High error on unseen validation/test data<br>"
            "• <strong>Noise Learning</strong>: Model memorizes noise and outliers instead of signal<br>"
            "• <strong>High Variance</strong>: Small data changes cause large model changes<br>"
            "• <strong>Complexity Symptom</strong>: Often from overly complex models relative to data<br>"
            "• <strong>Diagnostic Gap</strong>: Large gap between training and validation performance<br><br>"
            "Causes:<br>"
            "• Insufficient training data<br>"
            "• Excessive model complexity (too many parameters)<br>"
            "• Training for too many epochs<br>"
            "• Lack of regularization<br><br>"
            "Detection Methods:<br>"
            "• Learning curves (train vs validation)<br>"
            "• Cross-validation performance<br>"
            "• Early stopping monitoring<br><br>"
            "Prevention Techniques:<br>"
            "• Regularization (L1/L2, dropout)<br>"
            "• Data augmentation<br>"
            "• Feature selection<br>"
            "• Ensemble methods<br>"
            "• Early stopping"
        ),
        "category": "Model Behavior",
        "icon": "📊",
        "complexity": "Intermediate",
        "related_terms": ["Underfitting", "Bias-Variance Tradeoff", "Regularization", "Early Stopping"]
    },

    "Underfitting": {
        "definition": (
            "Underfitting occurs when a model is too simple to capture underlying patterns in the data.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Poor Training Performance</strong>: High error even on training data<br>"
            "• <strong>Poor Generalization</strong>: Also performs poorly on test data<br>"
            "• <strong>High Bias</strong>: Strong assumptions prevent learning true relationships<br>"
            "• <strong>Oversimplification</strong>: Model cannot represent necessary complexity<br>"
            "• <strong>Systematic Error</strong>: Consistent prediction errors across datasets<br>"
            "• <strong>Diagnostic Indicator</strong>: Both training and validation errors are high<br><br>"
            "Causes:<br>"
            "• Excessively simple model architecture<br>"
            "• Insufficient features or feature engineering<br>"
            "• Excessive regularization<br>"
            "• Training stopped too early<br><br>"
            "Detection Methods:<br>"
            "• Learning curve analysis<br>"
            "• Comparison with baseline models<br>"
            "• Residual analysis<br><br>"
            "Remediation Strategies:<br>"
            "• Increase model complexity<br>"
            "• Add relevant features<br>"
            "• Reduce regularization strength<br>"
            "• Train for more epochs<br>"
            "• Use ensemble methods"
        ),
        "category": "Model Behavior",
        "icon": "📉",
        "complexity": "Intermediate",
        "related_terms": ["Overfitting", "Bias-Variance Tradeoff", "Feature Engineering", "Model Complexity"]
    },

    "Bias (Statistical Bias)": {
        "definition": (
            "Bias represents error from erroneous assumptions in the learning algorithm.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Systematic Deviation</strong>: Consistent error in same direction<br>"
            "• <strong>Underfitting Driver</strong>: High bias leads to underfitting<br>"
            "• <strong>Assumption Error</strong>: Model assumptions don't match data reality<br>"
            "• <strong>Irreducible with Data</strong>: More data alone doesn't fix high bias<br>"
            "• <strong>Trade-off Component</strong>: Part of bias-variance trade-off<br>"
            "• <strong>Examples</strong>: Linear model for nonlinear relationship<br><br>"
            "Types of Bias:<br>"
            "• <strong>Algorithmic Bias</strong>: From model assumptions<br>"
            "• <strong>Selection Bias</strong>: From non-random training data<br>"
            "• <strong>Measurement Bias</strong>: From systematic measurement errors<br>"
            "• <strong>Confirmation Bias</strong>: From reinforcing existing beliefs<br><br>"
            "Mathematical Representation:<br>"
            "Bias² = E[(f̂(x) - f(x))²] where f(x) is true function, f̂(x) is estimate<br><br>"
            "Reduction Strategies:<br>"
            "• More flexible models<br>"
            "• Feature engineering<br>"
            "• Ensemble methods (boosting)<br>"
            "• Proper algorithm selection"
        ),
        "category": "Model Behavior",
        "icon": "⚡",
        "complexity": "Advanced",
        "related_terms": ["Variance", "Bias-Variance Tradeoff", "Underfitting", "Expected Error"]
    },

    "Variance": {
        "definition": (
            "Variance measures how much a model's predictions change with different training data.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Data Sensitivity</strong>: Model's sensitivity to specific training examples<br>"
            "• <strong>Overfitting Driver</strong>: High variance leads to overfitting<br>"
            "• <strong>Instability Indicator</strong>: Small data changes cause large prediction changes<br>"
            "• <strong>Complexity Correlation</strong>: Increases with model complexity<br>"
            "• <strong>Trade-off Component</strong>: Part of bias-variance trade-off<br>"
            "• <strong>Reducible with Data</strong>: More data typically reduces variance<br><br>"
            "Mathematical Representation:<br>"
            "Variance = E[(f̂(x) - E[f̂(x)])²] where f̂(x) is model prediction<br><br>"
            "Sources of High Variance:<br>"
            "• Too many parameters relative to data<br>"
            "• Complex nonlinear models<br>"
            "• Noisy training data<br>"
            "• Insufficient regularization<br><br>"
            "Reduction Strategies:<br>"
            "• More training data<br>"
            "• Regularization techniques<br>"
            "• Ensemble methods (bagging)<br>"
            "• Feature selection<br>"
            "• Simpler models<br><br>"
            "Expected Error Decomposition:<br>"
            "Expected Error = Bias² + Variance + Irreducible Error"
        ),
        "category": "Model Behavior",
        "icon": "🎲",
        "complexity": "Advanced",
        "related_terms": ["Bias", "Bias-Variance Tradeoff", "Overfitting", "Regularization"]
    },

    "Regularization": {
        "definition": (
            "Regularization techniques prevent overfitting by adding constraints to model parameters.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Overfitting Prevention</strong>: Primary defense against high variance<br>"
            "• <strong>Complexity Penalty</strong>: Adds penalty term to loss function<br>"
            "• <strong>Parameter Constraint</strong>: Encourages smaller, simpler parameters<br>"
            "• <strong>Generalization Improvement</strong>: Helps model generalize to unseen data<br>"
            "• <strong>Trade-off Management</strong>: Balances fit and complexity<br>"
            "• <strong>Universal Application</strong>: Used in linear models, neural networks, etc.<br><br>"
            "Common Regularization Techniques:<br>"
            "• <strong>L1 Regularization (Lasso)</strong>: Adds absolute value penalty, promotes sparsity<br>"
            "• <strong>L2 Regularization (Ridge)</strong>: Adds squared value penalty, smooths weights<br>"
            "• <strong>Elastic Net</strong>: Combines L1 and L2 regularization<br>"
            "• <strong>Dropout</strong>: Randomly ignores neurons during training (neural networks)<br>"
            "• <strong>Early Stopping</strong>: Stops training when validation error increases<br>"
            "• <strong>Data Augmentation</strong>: Artificially increases training data variety<br><br>"
            "Mathematical Form (L2):<br>"
            "Loss = Original Loss + λ × Σ(θᵢ²)<br>"
            "where λ is regularization strength hyperparameter<br><br>"
            "Hyperparameter Tuning:<br>"
            "Regularization strength (λ) is critical and requires careful cross-validation"
        ),
        "category": "Training Techniques",
        "icon": "🛡️",
        "complexity": "Intermediate",
        "related_terms": ["Overfitting", "L1 Regularization", "L2 Regularization", "Dropout", "Early Stopping"]
    },

    "Cross-Validation": {
        "definition": (
            "Cross-validation is a robust technique for assessing model performance on limited data.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Data Efficiency</strong>: Maximizes use of limited data for evaluation<br>"
            "• <strong>Performance Estimation</strong>: Provides unbiased estimate of generalization error<br>"
            "• <strong>Model Selection</strong>: Helps compare different models/algorithms<br>"
            "• <strong>Hyperparameter Tuning</strong>: Essential for tuning model parameters<br>"
            "• <strong>Variance Reduction</strong>: Multiple folds reduce evaluation variance<br>"
            "• <strong>Computational Cost</strong>: Increases training time k-fold<br><br>"
            "Common Cross-Validation Methods:<br>"
            "• <strong>k-Fold CV</strong>: Data split into k equal folds, each used as validation once<br>"
            "• <strong>Stratified k-Fold</strong>: Preserves class distribution in each fold<br>"
            "• <strong>Leave-One-Out (LOO)</strong>: Extreme case where k = n (computationally expensive)<br>"
            "• <strong>Time Series CV</strong>: Special methods for temporal data<br>"
            "• <strong>Nested CV</strong>: Outer loop for evaluation, inner loop for hyperparameter tuning<br><br>"
            "Best Practices:<br>"
            "• Use stratified CV for imbalanced data<br>"
            "• Shuffle data before splitting (except time series)<br>"
            "• Report mean and standard deviation of scores<br>"
            "• Ensure no data leakage between folds<br><br>"
            "Typical k Values:<br>"
            "• Small datasets: 5 or 10 folds<br>"
            "• Large datasets: 3 or 5 folds (computational constraints)<br>"
            "• Very small datasets: Leave-One-Out"
        ),
        "category": "Evaluation Techniques",
        "icon": "🔄",
        "complexity": "Intermediate",
        "related_terms": ["Train-Test Split", "Hyperparameter Tuning", "Overfitting", "Model Selection"]
    },

    "Feature Engineering": {
        "definition": (
            "Feature engineering transforms raw data into informative features that improve model performance.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Performance Critical</strong>: Often more impactful than algorithm choice<br>"
            "• <strong>Domain Knowledge Intensive</strong>: Requires understanding of problem context<br>"
            "• <strong>Iterative Process</strong>: Continuous refinement based on model feedback<br>"
            "• <strong>Creativity Required</strong>: Combines technical skill with insight<br>"
            "• <strong>Automation Potential</strong>: Automated feature engineering tools emerging<br>"
            "• <strong>Pipeline Essential</strong>: Core component of ML production pipelines<br><br>"
            "Common Feature Engineering Techniques:<br>"
            "• <strong>Encoding</strong>: Categorical to numerical (one-hot, label, target encoding)<br>"
            "• <strong>Scaling</strong>: Normalization, standardization for distance-based models<br>"
            "• <strong>Interaction Features</strong>: Multiplying/dividing existing features<br>"
            "• <strong>Polynomial Features</strong>: Creating squared/cubed terms for nonlinearity<br>"
            "• <strong>Binning</strong>: Converting continuous to categorical<br>"
            "• <strong>Date/Time Decomposition</strong>: Extracting day, month, season, etc.<br>"
            "• <strong>Text Features</strong>: TF-IDF, n-grams, embeddings<br>"
            "• <strong>Aggregation</strong>: Group statistics (mean, sum, count)<br><br>"
            "Best Practices:<br>"
            "• Start with domain understanding<br>"
            "• Create features that are interpretable<br>"
            "• Avoid target leakage<br>"
            "• Monitor feature importance<br>"
            "• Regularize to prevent overfitting from many features"
        ),
        "category": "Data Preparation",
        "icon": "🔧",
        "complexity": "Intermediate",
        "related_terms": ["Feature Selection", "Normalization", "Encoding", "Dimensionality Reduction"]
    },

    "Hyperparameter": {
        "definition": (
            "Hyperparameters are configuration settings that control the learning process and must be set before training.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>External Configuration</strong>: Set by data scientist/engineer, not learned from data<br>"
            "• <strong>Algorithm Control</strong>: Govern learning behavior and model capacity<br>"
            "• <strong>Performance Critical</strong>: Significantly impact final model quality<br>"
            "• <strong>Tuning Required</strong>: Optimal values found through systematic search<br>"
            "• <strong>Problem-Specific</strong>: Optimal values vary by dataset and problem<br>"
            "• <strong>Hierarchical</strong>: Some hyperparameters control others (e.g., network architecture)<br><br>"
            "Common Hyperparameters by Model Type:<br>"
            "• <strong>Neural Networks</strong>: Learning rate, batch size, layers, neurons, dropout rate<br>"
            "• <strong>Tree Models</strong>: Max depth, min samples split, number of estimators<br>"
            "• <strong>SVMs</strong>: C (regularization), kernel type, gamma<br>"
            "• <strong>k-NN</strong>: Number of neighbors, distance metric<br>"
            "• <strong>Regularization Models</strong>: Lambda/alpha strength<br><br>"
            "Hyperparameter Tuning Methods:<br>"
            "• <strong>Grid Search</strong>: Exhaustive search over predefined grid<br>"
            "• <strong>Random Search</strong>: Random sampling of hyperparameter space<br>"
            "• <strong>Bayesian Optimization</strong>: Probabilistic model-guided search<br>"
            "• <strong>Genetic Algorithms</strong>: Evolutionary search approach<br>"
            "• <strong>Hyperband</strong>: Adaptive resource allocation for tuning<br><br>"
            "Best Practices:<br>"
            "• Use cross-validation for evaluation<br>"
            "• Start with broad search, then refine<br>"
            "• Consider computational constraints<br>"
            "• Document tuning process and results"
        ),
        "category": "Model Configuration",
        "icon": "🎛️",
        "complexity": "Intermediate",
        "related_terms": ["Learning Rate", "Cross-Validation", "Grid Search", "Bayesian Optimization"]
    },

    "Learning Rate": {
        "definition": (
            "Learning rate controls the step size during gradient-based optimization.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Optimization Control</strong>: Most critical hyperparameter for gradient descent<br>"
            "• <strong>Step Size Determiner</strong>: How far to move in parameter space each update<br>"
            "• <strong>Convergence Affector</strong>: Directly impacts training stability and speed<br>"
            "• <strong>Problem-Sensitive</strong>: Optimal value varies by model and data<br>"
            "• <strong>Adaptive Variants</strong>: Modern optimizers adapt learning rate during training<br>"
            "• <strong>Schedule Potential</strong>: Can be decreased over time (learning rate decay)<br><br>"
            "Effects of Different Learning Rates:<br>"
            "• <strong>Too High</strong>: Divergence, oscillations, failure to converge<br>"
            "• <strong>Too Low</strong>: Slow convergence, risk of getting stuck in local minima<br>"
            "• <strong>Optimal</strong>: Steady decrease in loss, efficient convergence<br><br>"
            "Common Learning Rate Schedules:<br>"
            "• <strong>Constant</strong>: Fixed throughout training<br>"
            "• <strong>Step Decay</strong>: Reduce by factor after fixed epochs<br>"
            "• <strong>Exponential Decay</strong>: Continuous reduction<br>"
            "• <strong>Cosine Annealing</strong>: Smooth periodic reduction<br>"
            "• <strong>Cyclic</strong>: Oscillates between bounds (helps escape local minima)<br><br>"
            "Selection Guidelines:<br>"
            "• Typical range: 0.1 to 0.0001<br>"
            "• Start with 0.01 or 0.001 as baseline<br>"
            "• Use learning rate finder techniques<br>"
            "• Monitor loss curve for signs of instability<br><br>"
            "Mathematical Update Rule:<br>"
            "θ = θ - η × ∇J(θ)<br>"
            "where η is learning rate, ∇J(θ) is gradient"
        ),
        "category": "Optimization",
        "icon": "⏱️",
        "complexity": "Intermediate",
        "related_terms": ["Gradient Descent", "Optimizer", "Learning Rate Schedule", "Adam Optimizer"]
    },

    "Epoch": {
        "definition": (
            "An epoch represents one complete pass through the entire training dataset.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Training Unit</strong>: Fundamental measure of training progress<br>"
            "• <strong>Full Dataset Usage</strong>: Model sees every training example once per epoch<br>"
            "• <strong>Iterative Learning</strong>: Multiple epochs gradually improve performance<br>"
            "• <strong>Monitoring Metric</strong>: Primary x-axis for training curves<br>"
            "• <strong>Batch Relationship</strong>: One epoch contains multiple batches (batch updates)<br>"
            "• <strong>Early Stopping Basis</strong>: Training stopped based on epoch-wise validation performance<br><br>"
            "Epoch vs Iteration vs Batch:<br>"
            "• <strong>Batch</strong>: Subset of data used for one parameter update<br>"
            "• <strong>Iteration</strong>: One parameter update (processing one batch)<br>"
            "• <strong>Epoch</strong>: Number of iterations to process entire dataset<br><br>"
            "Calculating Iterations per Epoch:<br>"
            "Iterations per epoch = ceil(N / batch_size)<br>"
            "where N is total training samples<br><br>"
            "Epoch Strategy Considerations:<br>"
            "• <strong>Too Few Epochs</strong>: Underfitting, model hasn't learned enough<br>"
            "• <strong>Too Many Epochs</strong>: Overfitting, memorizes training data<br>"
            "• <strong>Early Stopping</strong>: Stop when validation performance plateaus/worsens<br><br>"
            "Typical Epoch Ranges:<br>"
            "• Simple models: 10-50 epochs<br>"
            "• Deep learning: 50-500+ epochs<br>"
            "• Large datasets: May use fewer epochs (data efficiency)<br><br>"
            "Monitoring During Training:<br>"
            "• Track training loss per epoch<br>"
            "• Monitor validation metrics<br>"
            "• Watch for divergence or plateaus"
        ),
        "category": "Training Process",
        "icon": "🔁",
        "complexity": "Beginner",
        "related_terms": ["Batch Size", "Iteration", "Early Stopping", "Training Curve"]
    },

    "Batch Size": {
        "definition": (
            "Batch size determines how many training examples are processed before updating model parameters.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Memory Trade-off</strong>: Larger batches require more memory<br>"
            "• <strong>Gradient Quality</strong>: Affects gradient estimate variance<br>"
            "• <strong>Convergence Speed</strong>: Impacts number of updates per epoch<br>"
            "• <strong>Hardware Utilization</strong>: Affects GPU/TPU efficiency<br>"
            "• <strong>Generalization Impact</strong>: Influences model's final performance<br>"
            "• <strong>Optimization Effect</strong>: Changes loss landscape traversal<br><br>"
            "Batch Size Spectrum:<br>"
            "• <strong>Batch Gradient Descent</strong>: batch_size = N (entire dataset)<br>"
            "• <strong>Mini-batch GD</strong>: 1 < batch_size < N (typical)<br>"
            "• <strong>Stochastic GD</strong>: batch_size = 1<br><br>"
            "Effects on Training:<br>"
            "• <strong>Small Batches</strong>:<br>"
            "  • Noisier gradients (regularization effect)<br>"
            "  • More updates per epoch<br>"
            "  • Better generalization often<br>"
            "  • Less memory required<br>"
            "• <strong>Large Batches</strong>:<br>"
            "  • Smoother gradients<br>"
            "  • Faster computation (hardware optimization)<br>"
            "  • Potential generalization issues<br>"
            "  • More memory required<br><br>"
            "Selection Guidelines:<br>"
            "• Start with 32 or 64 as baseline<br>"
            "• Power of 2 for hardware optimization (32, 64, 128, 256)<br>"
            "• Adjust based on memory constraints<br>"
            "• Consider learning rate scaling: when increasing batch size, often increase learning rate<br><br>"
            "Batch Size Heuristics:<br>"
            "• Small datasets: Smaller batches (16-64)<br>"
            "• Large datasets: Larger batches (256-1024)<br>"
            "• Deep learning: 32-256 common<br>"
            "• Transfer learning: Often use original model's batch size"
        ),
        "category": "Training Process",
        "icon": "📦",
        "complexity": "Intermediate",
        "related_terms": ["Epoch", "Gradient Descent", "Learning Rate", "Memory Management"]
    },

    "Gradient Descent": {
        "definition": (
            "Gradient descent is an iterative optimization algorithm for minimizing differentiable functions.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>First-Order Optimization</strong>: Uses first derivatives (gradients)<br>"
            "• <strong>Iterative Approach</strong>: Updates parameters repeatedly toward minimum<br>"
            "• <strong>Local Optimization</strong>: Finds local minima (not necessarily global)<br>"
            "• <strong>Foundation Algorithm</strong>: Basis for most neural network training<br>"
            "• <strong>Gradient Requirement</strong>: Requires differentiable loss function<br>"
            "• <strong>Backpropagation Partner</strong>: Gradient descent uses gradients computed via backpropagation<br><br>"
            "Core Update Rule:<br>"
            "θ = θ - η × ∇J(θ)<br>"
            "where θ are parameters, η is learning rate, ∇J(θ) is gradient of loss<br><br>"
            "Gradient Descent Variants:<br>"
            "• <strong>Batch Gradient Descent</strong>: Uses entire dataset per update<br>"
            "  • Pros: Stable convergence, deterministic<br>"
            "  • Cons: Slow for large datasets, memory intensive<br>"
            "• <strong>Stochastic GD (SGD)</strong>: Uses single sample per update<br>"
            "  • Pros: Fast updates, online learning possible<br>"
            "  • Cons: High variance, noisy convergence<br>"
            "• <strong>Mini-batch GD</strong>: Uses small batch per update (most common)<br>"
            "  • Pros: Balance of speed and stability<br>"
            "  • Cons: Introduces batch size hyperparameter<br><br>"
            "Advanced Optimizers (GD Extensions):<br>"
            "• <strong>Momentum</strong>: Accumulates velocity for faster convergence<br>"
            "• <strong>Adam</strong>: Adaptive moments (most popular for deep learning)<br>"
            "• <strong>RMSProp</strong>: Adaptive learning rate per parameter<br>"
            "• <strong>Adagrad</strong>: Adapts learning rate based on parameter history<br><br>"
            "Challenges and Solutions:<br>"
            "• <strong>Local Minima</strong>: Random initialization, momentum, restarts<br>"
            "• <strong>Vanishing Gradients</strong>: ReLU activation, batch normalization<br>"
            "• <strong>Learning Rate Selection</strong>: Learning rate schedules, adaptive methods"
        ),
        "category": "Optimization Algorithms",
        "icon": "⬇️",
        "complexity": "Intermediate",
        "related_terms": ["Learning Rate", "Backpropagation", "Optimizer", "Loss Function"]
    },

    "Loss Function (Cost Function)": {
        "definition": (
            "A loss function quantifies how poorly a model's predictions match the true values.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Optimization Target</strong>: What the training algorithm minimizes<br>"
            "• <strong>Differentiability Requirement</strong>: Must be differentiable for gradient-based methods<br>"
            "• <strong>Problem-Specific</strong>: Choice depends on task type<br>"
            "• <strong>Training Guide</strong>: Provides error signal for parameter updates<br>"
            "• <strong>Evaluation Metric Relation</strong>: Often related to but different from evaluation metrics<br>"
            "• <strong>Convexity Impact</strong>: Convex losses have single global minimum<br><br>"
            "Common Loss Functions by Task:<br>"
            "• <strong>Regression Tasks</strong>:<br>"
            "  • Mean Squared Error (MSE): Emphasizes large errors<br>"
            "  • Mean Absolute Error (MAE): Robust to outliers<br>"
            "  • Huber Loss: Combines MSE and MAE benefits<br>"
            "• <strong>Classification Tasks</strong>:<br>"
            "  • Binary Cross-Entropy: Binary classification standard<br>"
            "  • Categorical Cross-Entropy: Multiclass classification standard<br>"
            "  • Hinge Loss: Used in SVMs<br>"
            "• <strong>Specialized Tasks</strong>:<br>"
            "  • Triplet Loss: Metric learning, face recognition<br>"
            "  • Contrastive Loss: Siamese networks<br>"
            "  • Focal Loss: Addresses class imbalance<br><br>"
            "Mathematical Examples:<br>"
            "• MSE: (1/n) Σ(yᵢ - ŷᵢ)²<br>"
            "• Binary Cross-Entropy: -(1/n) Σ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]<br><br>"
            "Loss Function Selection Criteria:<br>"
            "• Task type (regression vs classification)<br>"
            "• Output distribution assumptions<br>"
            "• Robustness to outliers requirement<br>"
            "• Computational considerations<br>"
            "• Gradient behavior during training<br><br>"
            "Advanced Concepts:<br>"
            "• Custom loss functions for specific constraints<br>"
            "• Multi-task learning with combined losses<br>"
            "• Regularization terms added to loss"
        ),
        "category": "Optimization",
        "icon": "📐",
        "complexity": "Intermediate",
        "related_terms": ["Gradient Descent", "Optimizer", "Evaluation Metric", "Regularization"]
    },

    "Confusion Matrix": {
        "definition": (
            "A confusion matrix is a tabular visualization of classification model performance.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Error Analysis Tool</strong>: Reveals specific types of mistakes<br>"
            "• <strong>Multiclass Extension</strong>: Can be extended beyond binary classification<br>"
            "• <strong>Metric Foundation</strong>: Source for precision, recall, accuracy, F1<br>"
            "• <strong>Visual Diagnostic</strong>: Easy to interpret error patterns<br>"
            "• <strong>Imbalance Insight</strong>: Shows performance per class for imbalanced data<br>"
            "• <strong>Threshold Analysis</strong>: Changes with classification threshold<br><br>"
            "Binary Confusion Matrix Structure:<br>"
            "• <strong>True Positive (TP)</strong>: Correct positive predictions<br>"
            "• <strong>True Negative (TN)</strong>: Correct negative predictions<br>"
            "• <strong>False Positive (FP)</strong>: Negative incorrectly predicted as positive (Type I error)<br>"
            "• <strong>False Negative (FN)</strong>: Positive incorrectly predicted as negative (Type II error)<br><br>"
            "Derived Metrics from Confusion Matrix:<br>"
            "• Accuracy: (TP+TN) / Total<br>"
            "• Precision: TP / (TP+FP)<br>"
            "• Recall/Sensitivity: TP / (TP+FN)<br>"
            "• Specificity: TN / (TN+FP)<br>"
            "• F1 Score: Harmonic mean of precision and recall<br>"
            "• False Positive Rate: FP / (FP+TN)<br><br>"
            "Multiclass Confusion Matrix:<br>"
            "• Rows represent actual classes<br>"
            "• Columns represent predicted classes<br>"
            "• Diagonal shows correct predictions<br>"
            "• Off-diagonal shows confusion between classes<br><br>"
            "Advanced Analysis Techniques:<br>"
            "• Normalized confusion matrix (by row or column)<br>"
            "• Per-class metrics calculation<br>"
            "• Error pattern identification<br>"
            "• Threshold optimization using matrix changes<br><br>"
            "Visualization Best Practices:<br>"
            "• Use color scales for quick interpretation<br>"
            "• Include numerical values in cells<br>"
            "• Add marginal totals for context<br>"
            "• Consider logarithmic scale for large value ranges"
        ),
        "category": "Evaluation Tools",
        "icon": "🧮",
        "complexity": "Beginner",
        "related_terms": ["Precision", "Recall", "Accuracy", "ROC Curve", "Classification Report"]
    },

    "Train-Test Split": {
        "definition": (
            "Train-test split divides data into separate sets for model training and evaluation.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Generalization Assessment</strong>: Evaluates performance on unseen data<br>"
            "• <strong>Overfitting Detection</strong>: Reveals gap between training and test performance<br>"
            "• <strong>Simple Implementation</strong>: Easy to implement and understand<br>"
            "• <strong>Statistical Assumption</strong>: Assumes IID (independent and identically distributed) data<br>"
            "• <strong>Variance Concern</strong>: Single split can give variable performance estimates<br>"
            "• <strong>Data Leakage Prevention</strong>: Critical for avoiding overly optimistic estimates<br><br>"
            "Standard Split Ratios:<br>"
            "• 80/20: Common default (80% train, 20% test)<br>"
            "• 70/30: When more test data needed<br>"
            "• 90/10: When data is limited<br>"
            "• 60/20/20: With additional validation set<br><br>"
            "Split Methodologies:<br>"
            "• <strong>Random Split</strong>: Most common, assumes IID data<br>"
            "• <strong>Stratified Split</strong>: Preserves class distribution in both sets<br>"
            "• <strong>Time-Based Split</strong>: For temporal data (train on past, test on future)<br>"
            "• <strong>Grouped Split</strong>: Ensures same group doesn't appear in both sets<br>"
            "• <strong>Geographic Split</strong>: For spatial data independence<br><br>"
            "Best Practices:<br>"
            "• Perform split before any preprocessing<br>"
            "• Use stratification for imbalanced classes<br>"
            "• Set random seed for reproducibility<br>"
            "• Consider dataset size when choosing ratio<br>"
            "• Ensure no data leakage between sets<br><br>"
            "Limitations and Alternatives:<br>"
            "• <strong>Single split variance</strong>: Use cross-validation for more stable estimates<br>"
            "• <strong>Small datasets</strong>: Consider leave-one-out or bootstrap methods<br>"
            "• <strong>Temporal data</strong>: Use time series cross-validation<br><br>"
            "Implementation Considerations:<br>"
            "• sklearn.model_selection.train_test_split()<br>"
            "• stratification parameter for balanced splits<br>"
            "• shuffle parameter control<br>"
            "• random_state for reproducibility"
        ),
        "category": "Evaluation Techniques",
        "icon": "✂️",
        "complexity": "Beginner",
        "related_terms": ["Cross-Validation", "Overfitting", "Data Leakage", "Stratified Sampling"]
    },

    "Normalization (Min-Max Scaling)": {
        "definition": (
            "Normalization rescales features to a fixed range, typically [0, 1].<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Range Transformation</strong>: Maps original range to [0,1] or other fixed interval<br>"
            "• <strong>Distance Preservation</strong>: Maintains relative distances between values<br>"
            "• <strong>Outlier Sensitivity</strong>: Highly sensitive to extreme values<br>"
            "• <strong>Bounded Output</strong>: Result always within specified bounds<br>"
            "• <strong>Interpretability</strong>: All features have same scale for comparison<br>"
            "• <strong>Algorithm Suitability</strong>: Particularly useful for distance-based algorithms<br><br>"
            "Mathematical Formulation:<br>"
            "x' = (x - min(x)) / (max(x) - min(x))<br>"
            "Generalized to [a,b]: x' = a + (x - min(x)) × (b-a) / (max(x)-min(x))<br><br>"
            "When to Use Normalization:<br>"
            "• Neural networks (helps gradient descent)<br>"
            "• Distance-based algorithms (k-NN, k-means)<br>"
            "• Algorithms requiring bounded input<br>"
            "• When feature ranges vary significantly<br>"
            "• Image pixel data (natural [0,255] range)<br><br>"
            "Comparison with Standardization:<br>"
            "• <strong>Normalization</strong>: Bounded range, sensitive to outliers<br>"
            "• <strong>Standardization</strong>: Unbounded, more robust to outliers<br>"
            "• <strong>Choice depends</strong>: On algorithm and data characteristics<br><br>"
            "Practical Considerations:<br>"
            "• Compute min/max on training set only<br>"
            "• Apply same transformation to test data<br>"
            "• Handle constant features (division by zero)<br>"
            "• Consider robust min-max with percentiles for outlier handling<br><br>"
            "Implementation Example (scikit-learn):<br>"
            "from sklearn.preprocessing import MinMaxScaler<br>"
            "scaler = MinMaxScaler(feature_range=(0, 1))<br>"
            "X_train_scaled = scaler.fit_transform(X_train)<br>"
            "X_test_scaled = scaler.transform(X_test)<br><br>"
            "Alternative Normalization Methods:<br>"
            "• Unit Vector normalization (L2 normalization)<br>"
            "• Decimal scaling<br>"
            "• Robust scaling (using percentiles)"
        ),
        "category": "Data Preprocessing",
        "icon": "📏",
        "complexity": "Beginner",
        "related_terms": ["Standardization", "Feature Scaling", "Preprocessing", "Data Transformation"]
    },

    "Standardization (Z-score Normalization)": {
        "definition": (
            "Standardization transforms features to have zero mean and unit variance.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Gaussian Transformation</strong>: Assumes or creates approximately Gaussian distribution<br>"
            "• <strong>Outlier Robustness</strong>: More robust to outliers than min-max scaling<br>"
            "• <strong>Unbounded Output</strong>: Resulting values can exceed original range<br>"
            "• <strong>Statistical Foundation</strong>: Based on mean and standard deviation<br>"
            "• <strong>Algorithm Preference</strong>: Preferred by many linear models and SVMs<br>"
            "• <strong>Interpretation</strong>: Values represent number of standard deviations from mean<br><br>"
            "Mathematical Formulation:<br>"
            "x' = (x - μ) / σ<br>"
            "where μ is mean, σ is standard deviation<br><br>"
            "When to Use Standardization:<br>"
            "• Linear models (regression, logistic regression)<br>"
            "• Support Vector Machines<br>"
            "• Principal Component Analysis<br>"
            "• When data contains outliers<br>"
            "• When algorithm assumes standardized features<br><br>"
            "Statistical Properties:<br>"
            "• Transformed features have mean = 0<br>"
            "• Transformed features have variance = 1<br>"
            "• Preserves shape of original distribution<br>"
            "• Maintains relationships between features<br><br>"
            "Practical Implementation Considerations:<br>"
            "• Fit scaler only on training data<br>"
            "• Transform both training and test with same parameters<br>"
            "• Handle near-constant features (small σ)<br>"
            "• Consider robust standardization (median/IQR) for outliers<br><br>"
            "Comparison with Normalization:<br>"
            "• <strong>Standardization</strong>: Better for outliers, unbounded, preserves distribution<br>"
            "• <strong>Normalization</strong>: Bounded range, sensitive to outliers, changes distribution shape<br><br>"
            "Advanced Variants:<br>"
            "• RobustScaler: Uses median and IQR instead of mean/std<br>"
            "• QuantileTransformer: Maps to uniform/Gaussian distribution<br>"
            "• PowerTransformer: Applies power transforms to normalize"
        ),
        "category": "Data Preprocessing",
        "icon": "📊",
        "complexity": "Beginner",
        "related_terms": ["Normalization", "Gaussian Distribution", "Feature Scaling", "Preprocessing"]
    },

    "Ensemble Methods": {
        "definition": (
            "Ensemble methods combine multiple models to produce better predictions than any single model.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Wisdom of Crowds</strong>: Leverages multiple perspectives<br>"
            "• <strong>Error Reduction</strong>: Reduces variance and/or bias<br>"
            "• <strong>Robustness</strong>: More stable predictions<br>"
            "• <strong>Performance</strong>: Often top performers in competitions<br>"
            "• <strong>Complexity Cost</strong>: Increased computational requirements<br>"
            "• <strong>Interpretability Challenge</strong>: Harder to explain than single models<br><br>"
            "Core Ensemble Strategies:<br>"
            "• <strong>Bagging (Bootstrap Aggregating)</strong>:<br>"
            "  • Trains multiple models on different data subsets<br>"
            "  • Reduces variance<br>"
            "  • Examples: Random Forest, Extra Trees<br>"
            "• <strong>Boosting</strong>:<br>"
            "  • Sequentially trains models focusing on previous errors<br>"
            "  • Reduces bias<br>"
            "  • Examples: AdaBoost, Gradient Boosting, XGBoost<br>"
            "• <strong>Stacking</strong>:<br>"
            "  • Uses meta-model to combine base model predictions<br>"
            "  • Can combine different algorithm types<br>"
            "  • Powerful but complex<br>"
            "• <strong>Voting/Averaging</strong>:<br>"
            "  • Simple combination of predictions<br>"
            "  • Hard voting (classification) or soft voting (probabilities)<br><br>"
            "Key Ensemble Algorithms:<br>"
            "• <strong>Random Forest</strong>: Bagging of decision trees with feature randomness<br>"
            "• <strong>Gradient Boosting Machines</strong>: Sequential tree building minimizing loss gradient<br>"
            "• <strong>XGBoost</strong>: Optimized gradient boosting with regularization<br>"
            "• <strong>LightGBM</strong>: Gradient boosting with leaf-wise tree growth<br>"
            "• <strong>CatBoost</strong>: Gradient boosting optimized for categorical features<br><br>"
            "Ensemble Design Principles:<br>"
            "• <strong>Diversity</strong>: Base models should make different errors<br>"
            "• <strong>Competence</strong>: Each model should be reasonably accurate<br>"
            "• <strong>Combination Strategy</strong>: How to aggregate predictions effectively<br><br>"
            "When to Use Ensembles:<br>"
            "• When maximum accuracy is required<br>"
            "• When computational resources allow<br>"
            "• For competition settings<br>"
            "• In production when stability is critical"
        ),
        "category": "Modeling Techniques",
        "icon": "🤝",
        "complexity": "Intermediate",
        "related_terms": ["Bagging", "Boosting", "Random Forest", "XGBoost", "Model Aggregation"]
    },

    "Clustering": {
        "definition": (
            "Clustering groups similar data points together without using predefined labels.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Unsupervised Learning</strong>: No target variable required<br>"
            "• <strong>Exploratory Analysis</strong>: Discovers natural groupings in data<br>"
            "• <strong>Similarity-Based</strong>: Groups based on distance or density<br>"
            "• <strong>Dimensionality Tool</strong>: Can reveal structure in high-dimensional data<br>"
            "• <strong>Preprocessing Step</strong>: Sometimes used before supervised learning<br>"
            "• <strong>Validation Challenge</strong>: No ground truth for evaluation<br><br>"
            "Major Clustering Approaches:<br>"
            "• <strong>Centroid-Based</strong>: Groups around central points<br>"
            "  • Examples: K-Means, K-Medoids<br>"
            "  • Pros: Simple, efficient<br>"
            "  • Cons: Assumes spherical clusters, sensitive to initialization<br>"
            "• <strong>Density-Based</strong>: Groups based on density regions<br>"
            "  • Examples: DBSCAN, OPTICS<br>"
            "  • Pros: Handles arbitrary shapes, identifies outliers<br>"
            "  • Cons: Sensitive to parameters, struggles with varying densities<br>"
            "• <strong>Hierarchical</strong>: Creates nested cluster tree<br>"
            "  • Examples: Agglomerative, Divisive<br>"
            "  • Pros: No need to specify k, dendrogram visualization<br>"
            "  • Cons: Computationally expensive, irreversible merges/splits<br>"
            "• <strong>Distribution-Based</strong>: Assumes data from probability distributions<br>"
            "  • Examples: Gaussian Mixture Models<br>"
            "  • Pros: Soft clustering, probabilistic membership<br>"
            "  • Cons: Assumes distribution type<br><br>"
            "Clustering Evaluation Metrics:<br>"
            "• <strong>Internal</strong>: Based on data itself (silhouette score, Davies-Bouldin)<br>"
            "• <strong>External</strong>: When ground truth available (adjusted Rand index, NMI)<br>"
            "• <strong>Relative</strong>: Compare different clusterings<br><br>"
            "Practical Considerations:<br>"
            "• Feature scaling is usually necessary<br>"
            "• Distance metric choice is critical<br>"
            "• Determining optimal k is challenging (elbow method, silhouette analysis)<br>"
            "• Interpret and validate clusters with domain knowledge<br><br>"
            "Common Applications:<br>"
            "• Customer segmentation<br>"
            "• Image segmentation<br>"
            "• Document grouping<br>"
            "• Anomaly detection<br>"
            "• Social network analysis"
        ),
        "category": "Unsupervised Learning",
        "icon": "🎨",
        "complexity": "Intermediate",
        "related_terms": ["K-Means", "DBSCAN", "Hierarchical Clustering", "Dimensionality Reduction"]
    },

    "Dimensionality Reduction": {
        "definition": (
            "Dimensionality reduction transforms high-dimensional data into lower-dimensional representation.<br><br>"
            "Key Characteristics:<br>"
            "• <strong>Curse Mitigation</strong>: Addresses curse of dimensionality<br>"
            "• <strong>Visualization Enabler</strong>: Allows 2D/3D visualization of high-D data<br>"
            "• <strong>Noise Reduction</strong>: Often removes noisy or redundant dimensions<br>"
            "• <strong>Efficiency Improver</strong>: Speeds up training and inference<br>"
            "• <strong>Structure Revealer</strong>: Can uncover hidden patterns<br>"
            "• <strong>Information Trade-off</strong>: Balances compression with information preservation<br><br>"
            "Primary Dimensionality Reduction Techniques:<br>"
            "• <strong>Linear Methods</strong>:<br>"
            "  • Principal Component Analysis (PCA): Orthogonal linear projection maximizing variance<br>"
            "  • Linear Discriminant Analysis (LDA): Supervised method maximizing class separation<br>"
            "  • Factor Analysis: Models observed variables with fewer latent factors<br>"
            "• <strong>Nonlinear Methods</strong>:<br>"
            "  • t-SNE: Preserves local structure, excellent for visualization<br>"
            "  • UMAP: Preserves both local and global structure<br>"
            "  • Autoencoders: Neural network-based compression<br>"
            "  • Isomap: Preserves geodesic distances<br>"
            "  • LLE: Locally linear embedding<br><br>"
            "Selection Criteria:<br>"
            "• <strong>PCA</strong>: When linear relationships dominate, for decorrelation<br>"
            "• <strong>t-SNE</strong>: For visualization, exploring local structure<br>"
            "• <strong>UMAP</strong>: For visualization with better global structure preservation<br>"
            "• <strong>Autoencoders</strong>: When nonlinear relationships are complex<br>"
            "• <strong>LDA</strong>: When class labels are available and separation is goal<br><br>"
            "Practical Considerations:<br>"
            "• Scale features before linear methods<br>"
            "• Determine optimal number of components (scree plot, cumulative variance)<br>"
            "• t-SNE parameters (perplexity) significantly affect results<br>"
            "• UMAP generally faster and more scalable than t-SNE<br>"
            "• Reconstruction error for autoencoder evaluation<br><br>"
            "Applications:<br>"
            "• Data visualization (exploratory analysis)<br>"
            "• Feature extraction for downstream tasks<br>"
            "• Data compression and storage<br>"
            "• Noise filtering<br>"
            "• Overcoming multicollinearity in regression<br><br>"
            "Limitations:<br>"
            "• Information loss inevitable<br>"
            "• Interpretability of reduced dimensions can be challenging<br>"
            "• Some methods computationally intensive<br>"
            "• Nonlinear methods may not preserve all relationships"
        ),
        "category": "Feature Engineering",
        "icon": "🗜️",
        "complexity": "Advanced",
        "related_terms": ["PCA", "t-SNE", "UMAP", "Autoencoder", "Feature Extraction"]
    }
}