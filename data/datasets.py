"""
Synthetic Data Generators for ML Portfolio
Provides realistic datasets for all algorithm demonstrations
"""
import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_regression, make_classification, make_blobs,
    make_moons, make_circles, make_swiss_roll
)
from sklearn.preprocessing import StandardScaler


# ============================================
# SAMPLE DATASETS REGISTRY
# ============================================
SAMPLE_DATASETS = {
    "house_prices": {
        "name": "🏠 House Prices",
        "description": "Predict house prices based on features like sqft, bedrooms, bathrooms, age, and location score.",
        "task": "regression",
        "target": "price",
        "features": ["sqft", "bedrooms", "bathrooms", "age", "location_score"],
        "samples": 500,
        "generator": "generate_house_prices"
    },
    "spam_detection": {
        "name": "📧 Email Spam",
        "description": "Classify emails as spam or not spam based on word frequencies and formatting features.",
        "task": "classification",
        "target": "is_spam",
        "features": ["word_freq_free", "word_freq_money", "word_freq_win", "capital_run_length", "exclamation_count"],
        "samples": 1000,
        "generator": "generate_spam_data"
    },
    "student_performance": {
        "name": "📚 Student Performance",
        "description": "Predict exam scores based on study habits, attendance, and previous performance.",
        "task": "regression",
        "target": "exam_score",
        "features": ["study_hours", "homework_hours", "class_attendance", "sleep_hours", "previous_gpa"],
        "samples": 400,
        "generator": "generate_student_data"
    },
    "customer_churn": {
        "name": "👤 Customer Churn",
        "description": "Predict whether a customer will churn based on usage and engagement metrics.",
        "task": "classification",
        "target": "churned",
        "features": ["tenure", "monthly_charges", "total_charges", "support_tickets"],
        "samples": 500,
        "generator": "generate_churn_data"
    },
    "loan_approval": {
        "name": "💳 Loan Approval",
        "description": "Predict loan approval based on income, credit score, and debt ratios.",
        "task": "classification",
        "target": "approved",
        "features": ["income", "debt", "credit_score", "employment_years", "loan_amount"],
        "samples": 500,
        "generator": "generate_loan_data"
    },
    "car_prices": {
        "name": "🚗 Car Prices",
        "description": "Predict used car prices based on brand, mileage, age, and engine specs.",
        "task": "regression",
        "target": "price",
        "features": ["brand", "fuel_type", "year", "mileage", "engine_size", "horsepower"],
        "samples": 500,
        "generator": "generate_car_prices"
    },
    "fraud_detection": {
        "name": "🚨 Fraud Detection",
        "description": "Detect fraudulent transactions based on amount, time, and location patterns.",
        "task": "classification",
        "target": "is_fraud",
        "features": ["amount", "time_of_day", "distance_from_home"],
        "samples": 5000,
        "generator": "generate_fraud_data"
    },
    "customer_segments": {
        "name": "👥 Customer Segments",
        "description": "Customer segmentation data for clustering based on age, income, and spending.",
        "task": "clustering",
        "target": None,
        "features": ["age", "annual_income", "spending_score"],
        "samples": 500,
        "generator": "generate_customer_segments"
    },
    "stock_prices": {
        "name": "📈 Stock Prices",
        "description": "Stock price prediction with technical indicators like moving averages and volume.",
        "task": "regression",
        "target": "close_price",
        "features": ["volatility", "volume", "ma_5", "ma_20", "momentum"],
        "samples": 500,
        "generator": "generate_stock_data"
    },
    "parkinsons": {
        "name": "🩺 Parkinson's Detection",
        "description": "Detect Parkinson's disease based on voice signal measurements.",
        "task": "classification",
        "target": "has_parkinsons",
        "features": ["jitter", "shimmer", "hnr", "rpde", "dfa", "spread1"],
        "samples": 300,
        "generator": "generate_parkinsons_data"
    }
}


class DatasetGenerator:
    """Generate synthetic datasets for ML demonstrations"""
    
    @staticmethod
    def generate_house_prices(n_samples=500, noise=10, random_state=42):
        """Generate house price dataset for Linear Regression"""
        np.random.seed(random_state)
        
        # Features
        sqft = np.random.uniform(800, 4000, n_samples)
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples)
        age = np.random.uniform(0, 50, n_samples)
        location_score = np.random.uniform(1, 10, n_samples)
        
        # Price calculation (realistic formula)
        price = (
            50000 +
            150 * sqft +
            20000 * bedrooms +
            15000 * bathrooms -
            1000 * age +
            30000 * location_score +
            np.random.normal(0, noise * 1000, n_samples)
        )
        
        df = pd.DataFrame({
            'sqft': sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': age,
            'location_score': location_score,
            'price': price
        })
        
        return df
    
    @staticmethod
    def generate_marketing_data(n_samples=300, n_channels=8, random_state=42):
        """Generate marketing channel data for Lasso Regression"""
        np.random.seed(random_state)
        
        channels = ['TV', 'Radio', 'Social Media', 'Email', 'SEO', 
                    'PPC', 'Influencer', 'Print']
        
        # Generate spending data with varying importance
        importance = [0.3, 0.15, 0.25, 0.1, 0.2, 0, 0, 0]  # Some have no impact
        
        data = {}
        for i, channel in enumerate(channels):
            data[channel] = np.random.uniform(1000, 50000, n_samples)
        
        # Sales based on important channels only
        sales = 50000
        for i, channel in enumerate(channels):
            sales = sales + importance[i] * data[channel]
        
        sales = sales + np.random.normal(0, 5000, n_samples)
        data['Sales'] = sales
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_student_data(n_samples=400, random_state=42):
        """Generate student performance data for Ridge Regression"""
        np.random.seed(random_state)
        
        # Correlated features (multicollinearity)
        study_hours = np.random.uniform(1, 10, n_samples)
        homework_hours = 0.7 * study_hours + np.random.normal(0, 1, n_samples)
        class_attendance = 0.6 * study_hours + np.random.uniform(50, 100, n_samples)
        
        sleep_hours = np.random.uniform(4, 10, n_samples)
        stress_level = 10 - 0.5 * sleep_hours + np.random.normal(0, 1, n_samples)
        
        previous_gpa = np.random.uniform(2.0, 4.0, n_samples)
        
        # Exam score
        exam_score = (
            20 +
            5 * study_hours +
            3 * homework_hours +
            0.1 * class_attendance +
            2 * sleep_hours -
            1 * stress_level +
            10 * previous_gpa +
            np.random.normal(0, 5, n_samples)
        )
        exam_score = np.clip(exam_score, 0, 100)
        
        return pd.DataFrame({
            'study_hours': study_hours,
            'homework_hours': homework_hours,
            'class_attendance': class_attendance,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'previous_gpa': previous_gpa,
            'exam_score': exam_score
        })
    
    @staticmethod
    def generate_car_prices(n_samples=500, random_state=42):
        """Generate car price data for Elastic Net"""
        np.random.seed(random_state)
        
        # Categorical: brand, fuel_type
        brands = np.random.choice(['Toyota', 'Honda', 'BMW', 'Mercedes', 'Ford'], n_samples)
        fuel_types = np.random.choice(['Petrol', 'Diesel', 'Electric', 'Hybrid'], n_samples)
        
        # Continuous features
        year = np.random.randint(2010, 2024, n_samples)
        mileage = np.random.uniform(0, 150000, n_samples)
        engine_size = np.random.uniform(1.0, 5.0, n_samples)
        horsepower = 50 + 50 * engine_size + np.random.normal(0, 20, n_samples)
        
        # Brand multipliers
        brand_mult = {'Toyota': 1.0, 'Honda': 1.05, 'BMW': 1.8, 'Mercedes': 2.0, 'Ford': 0.9}
        fuel_mult = {'Petrol': 1.0, 'Diesel': 1.1, 'Electric': 1.3, 'Hybrid': 1.2}
        
        # Price calculation
        base_price = 15000
        price = base_price * np.array([brand_mult[b] for b in brands])
        price = price * np.array([fuel_mult[f] for f in fuel_types])
        price = price + 500 * (year - 2010)
        price = price - 0.05 * mileage
        price = price + 2000 * engine_size
        price = price + np.random.normal(0, 2000, n_samples)
        price = np.maximum(price, 5000)
        
        return pd.DataFrame({
            'brand': brands,
            'fuel_type': fuel_types,
            'year': year,
            'mileage': mileage,
            'engine_size': engine_size,
            'horsepower': horsepower,
            'price': price
        })
    
    @staticmethod
    def generate_stock_data(n_samples=500, random_state=42):
        """Generate stock price data for SVR"""
        np.random.seed(random_state)
        
        # Time series features
        t = np.arange(n_samples)
        
        # Technical indicators
        volatility = 0.02 + 0.01 * np.sin(2 * np.pi * t / 50)
        momentum = np.cumsum(np.random.normal(0, 1, n_samples))
        volume = np.random.uniform(1e6, 1e8, n_samples)
        
        # Moving averages
        ma_5 = pd.Series(momentum).rolling(5).mean().fillna(0).values
        ma_20 = pd.Series(momentum).rolling(20).mean().fillna(0).values
        
        # Price with nonlinear patterns
        price = 100 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 100) + momentum + volatility * 100
        price = np.maximum(price, 10)
        
        return pd.DataFrame({
            'day': t,
            'volatility': volatility,
            'volume': volume,
            'ma_5': ma_5,
            'ma_20': ma_20,
            'momentum': momentum,
            'close_price': price
        })
    
    @staticmethod
    def generate_battery_data(n_samples=200, random_state=42):
        """Generate battery degradation data for Polynomial Regression"""
        np.random.seed(random_state)
        
        # Cycles (0 to 1000)
        cycles = np.sort(np.random.uniform(0, 1000, n_samples))
        
        # Battery capacity follows polynomial decay
        capacity = (
            100 -
            0.01 * cycles -
            0.00005 * cycles**2 +
            0.00000001 * cycles**3 +
            np.random.normal(0, 2, n_samples)
        )
        capacity = np.clip(capacity, 20, 100)
        
        return pd.DataFrame({
            'cycles': cycles,
            'capacity': capacity
        })
    
    @staticmethod
    def generate_spam_data(n_samples=1000, random_state=42):
        """Generate email spam detection data for Logistic Regression"""
        np.random.seed(random_state)
        
        # Word frequency features
        features = {
            'word_freq_free': np.random.exponential(0.5, n_samples),
            'word_freq_money': np.random.exponential(0.3, n_samples),
            'word_freq_win': np.random.exponential(0.2, n_samples),
            'word_freq_urgent': np.random.exponential(0.4, n_samples),
            'word_freq_click': np.random.exponential(0.3, n_samples),
            'capital_run_length': np.random.exponential(5, n_samples),
            'capital_run_total': np.random.randint(1, 500, n_samples),
            'exclamation_count': np.random.poisson(2, n_samples)
        }
        
        # Spam probability
        spam_score = (
            0.3 * features['word_freq_free'] +
            0.4 * features['word_freq_money'] +
            0.3 * features['word_freq_win'] +
            0.2 * features['word_freq_urgent'] +
            0.01 * features['capital_run_length'] +
            0.05 * features['exclamation_count']
        )
        
        is_spam = (spam_score + np.random.normal(0, 0.3, n_samples)) > 0.5
        features['is_spam'] = is_spam.astype(int)
        
        return pd.DataFrame(features)
    
    @staticmethod
    def generate_movie_data(n_samples=500, n_users=100, random_state=42):
        """Generate movie rating data for KNN"""
        np.random.seed(random_state)
        
        # User preferences (latent factors)
        n_genres = 5
        user_prefs = np.random.randn(n_users, n_genres)
        movie_genres = np.random.randn(n_samples, n_genres)
        
        # Generate ratings
        ratings_matrix = np.dot(user_prefs, movie_genres.T)
        ratings_matrix = (ratings_matrix - ratings_matrix.min()) / (ratings_matrix.max() - ratings_matrix.min())
        ratings_matrix = 1 + 4 * ratings_matrix  # Scale to 1-5
        ratings_matrix = np.round(ratings_matrix * 2) / 2  # Round to nearest 0.5
        
        return pd.DataFrame(ratings_matrix, columns=[f'Movie_{i}' for i in range(n_samples)])
    
    @staticmethod
    def generate_parkinsons_data(n_samples=300, random_state=42):
        """Generate Parkinson's disease voice data for SVM"""
        np.random.seed(random_state)
        
        # Voice signal features
        has_parkinsons = np.random.binomial(1, 0.4, n_samples)
        
        # Features with different distributions for each class
        jitter = np.where(has_parkinsons, 
                         np.random.normal(0.006, 0.002, n_samples),
                         np.random.normal(0.003, 0.001, n_samples))
        
        shimmer = np.where(has_parkinsons,
                          np.random.normal(0.04, 0.015, n_samples),
                          np.random.normal(0.02, 0.008, n_samples))
        
        hnr = np.where(has_parkinsons,
                       np.random.normal(18, 4, n_samples),
                       np.random.normal(24, 3, n_samples))
        
        rpde = np.where(has_parkinsons,
                        np.random.normal(0.55, 0.1, n_samples),
                        np.random.normal(0.4, 0.08, n_samples))
        
        dfa = np.where(has_parkinsons,
                       np.random.normal(0.72, 0.05, n_samples),
                       np.random.normal(0.65, 0.04, n_samples))
        
        spread1 = np.where(has_parkinsons,
                          np.random.normal(-5, 1, n_samples),
                          np.random.normal(-6.5, 0.8, n_samples))
        
        return pd.DataFrame({
            'jitter': jitter,
            'shimmer': shimmer,
            'hnr': hnr,
            'rpde': rpde,
            'dfa': dfa,
            'spread1': spread1,
            'has_parkinsons': has_parkinsons
        })
    
    @staticmethod
    def generate_review_data(n_samples=500, random_state=42):
        """Generate sentiment analysis data for Naive Bayes"""
        np.random.seed(random_state)
        
        # Sentiment: 0=negative, 1=neutral, 2=positive
        sentiment = np.random.choice([0, 1, 2], n_samples, p=[0.25, 0.25, 0.5])
        
        # Word counts with sentiment correlation
        positive_words = np.where(sentiment == 2,
                                  np.random.poisson(10, n_samples),
                                  np.random.poisson(2, n_samples))
        
        negative_words = np.where(sentiment == 0,
                                  np.random.poisson(10, n_samples),
                                  np.random.poisson(2, n_samples))
        
        neutral_words = np.random.poisson(5, n_samples)
        
        exclamations = np.where(sentiment != 1,
                                np.random.poisson(3, n_samples),
                                np.random.poisson(1, n_samples))
        
        word_count = np.random.randint(20, 200, n_samples)
        
        return pd.DataFrame({
            'positive_word_count': positive_words,
            'negative_word_count': negative_words,
            'neutral_word_count': neutral_words,
            'exclamation_count': exclamations,
            'total_word_count': word_count,
            'sentiment': sentiment
        })
    
    @staticmethod
    def generate_digit_data(n_samples=500, random_state=42):
        """Generate simple digit classification data for Perceptron"""
        from sklearn.datasets import load_digits
        
        digits = load_digits()
        X = digits.data[:n_samples]
        y = digits.target[:n_samples]
        
        # Binary: odd vs even
        y_binary = y % 2
        
        return pd.DataFrame(X), pd.Series(y_binary, name='is_odd')
    
    @staticmethod
    def generate_loan_data(n_samples=500, random_state=42):
        """Generate loan approval data for Decision Tree"""
        np.random.seed(random_state)
        
        income = np.random.uniform(20000, 200000, n_samples)
        debt = np.random.uniform(0, 100000, n_samples)
        credit_score = np.random.randint(300, 850, n_samples)
        employment_years = np.random.uniform(0, 30, n_samples)
        loan_amount = np.random.uniform(5000, 500000, n_samples)
        
        # Approval logic
        debt_to_income = debt / income
        loan_to_income = loan_amount / income
        
        approval = (
            (credit_score > 600) &
            (debt_to_income < 0.4) &
            (loan_to_income < 5) &
            (employment_years > 1)
        ).astype(int)
        
        # Add some noise
        flip_mask = np.random.random(n_samples) < 0.05
        approval[flip_mask] = 1 - approval[flip_mask]
        
        return pd.DataFrame({
            'income': income,
            'debt': debt,
            'credit_score': credit_score,
            'employment_years': employment_years,
            'loan_amount': loan_amount,
            'debt_to_income': debt_to_income,
            'approved': approval
        })
    
    @staticmethod
    def generate_churn_data(n_samples=500, random_state=42):
        """Generate customer churn data for Random Forest"""
        np.random.seed(random_state)
        
        tenure = np.random.uniform(0, 72, n_samples)  # months
        monthly_charges = np.random.uniform(20, 100, n_samples)
        total_charges = monthly_charges * tenure
        
        contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples)
        payment_method = np.random.choice(['Credit card', 'Bank transfer', 'Electronic check'], n_samples)
        
        num_support_tickets = np.random.poisson(2, n_samples)
        has_streaming = np.random.binomial(1, 0.5, n_samples)
        has_phone = np.random.binomial(1, 0.7, n_samples)
        
        # Churn probability
        churn_prob = 0.2
        churn_prob = churn_prob - 0.003 * tenure
        churn_prob = churn_prob + 0.002 * monthly_charges
        churn_prob = churn_prob + np.where(contract_type == 'Month-to-month', 0.15, 0)
        churn_prob = churn_prob + 0.02 * num_support_tickets
        churn_prob = np.clip(churn_prob, 0.05, 0.9)
        
        churned = np.random.binomial(1, churn_prob)
        
        return pd.DataFrame({
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract_type': contract_type,
            'payment_method': payment_method,
            'support_tickets': num_support_tickets,
            'has_streaming': has_streaming,
            'has_phone': has_phone,
            'churned': churned
        })
    
    @staticmethod
    def generate_fraud_data(n_samples=5000, fraud_ratio=0.02, random_state=42):
        """Generate credit card fraud data for Gradient Boosting"""
        np.random.seed(random_state)
        
        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud
        
        # Normal transactions
        normal_amount = np.random.exponential(50, n_normal)
        normal_time = np.random.uniform(0, 24, n_normal)
        normal_distance = np.random.exponential(10, n_normal)
        
        # Fraud transactions
        fraud_amount = np.random.exponential(500, n_fraud)
        fraud_time = np.random.choice([2, 3, 4, 5], n_fraud) + np.random.normal(0, 0.5, n_fraud)
        fraud_distance = np.random.exponential(100, n_fraud)
        
        amounts = np.concatenate([normal_amount, fraud_amount])
        times = np.concatenate([normal_time, fraud_time])
        distances = np.concatenate([normal_distance, fraud_distance])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
        
        # Add more features
        hour_sin = np.sin(2 * np.pi * times / 24)
        hour_cos = np.cos(2 * np.pi * times / 24)
        
        # Shuffle
        idx = np.random.permutation(n_samples)
        
        return pd.DataFrame({
            'amount': amounts[idx],
            'time_of_day': times[idx],
            'distance_from_home': distances[idx],
            'hour_sin': hour_sin[idx],
            'hour_cos': hour_cos[idx],
            'is_fraud': labels[idx].astype(int)
        })
    
    @staticmethod
    def generate_customer_segments(n_samples=500, n_clusters=5, random_state=42):
        """Generate customer segmentation data for K-Means"""
        np.random.seed(random_state)
        
        # Generate clusters with different characteristics
        cluster_centers = [
            [50, 5000, 100],    # Low value, low spend
            [35, 50000, 500],   # Young professionals
            [55, 100000, 1000], # High value customers
            [25, 20000, 200],   # Students
            [45, 80000, 800]    # Middle-aged affluent
        ]
        
        X, _ = make_blobs(n_samples=n_samples, centers=cluster_centers,
                          cluster_std=[5, 5, 5, 5, 5], random_state=random_state)
        
        return pd.DataFrame({
            'age': np.clip(X[:, 0], 18, 80),
            'annual_income': np.clip(X[:, 1], 10000, 200000),
            'spending_score': np.clip(X[:, 2], 1, 2000)
        })
    
    @staticmethod
    def generate_network_traffic(n_samples=1000, random_state=42):
        """Generate network traffic data for DBSCAN"""
        np.random.seed(random_state)
        
        # Normal traffic clusters
        normal1 = np.random.multivariate_normal([50, 100], [[10, 0], [0, 20]], 400)
        normal2 = np.random.multivariate_normal([150, 80], [[15, 0], [0, 10]], 300)
        normal3 = np.random.multivariate_normal([100, 200], [[8, 0], [0, 30]], 250)
        
        # Anomalies (scattered)
        anomalies = np.random.uniform([0, 0], [200, 300], (50, 2))
        
        X = np.vstack([normal1, normal2, normal3, anomalies])
        
        return pd.DataFrame({
            'packets_per_second': X[:, 0],
            'bytes_per_packet': X[:, 1]
        })
    
    @staticmethod
    def generate_song_features(n_samples=300, random_state=42):
        """Generate song audio features for Hierarchical Clustering"""
        np.random.seed(random_state)
        
        # Genre-based clusters
        genres = ['Pop', 'Rock', 'Jazz', 'Classical', 'Electronic']
        genre_labels = np.random.choice(genres, n_samples)
        
        # Features vary by genre
        tempo = np.random.normal(120, 20, n_samples)
        energy = np.random.uniform(0.3, 0.9, n_samples)
        danceability = np.random.uniform(0.2, 0.9, n_samples)
        acousticness = np.random.uniform(0.1, 0.9, n_samples)
        valence = np.random.uniform(0.1, 0.9, n_samples)
        
        # Adjust by genre
        genre_mods = {
            'Pop': {'tempo': 10, 'danceability': 0.2},
            'Rock': {'energy': 0.2, 'acousticness': -0.3},
            'Jazz': {'acousticness': 0.3, 'tempo': -20},
            'Classical': {'acousticness': 0.4, 'energy': -0.3},
            'Electronic': {'energy': 0.2, 'danceability': 0.3}
        }
        
        return pd.DataFrame({
            'tempo': tempo,
            'energy': energy,
            'danceability': danceability,
            'acousticness': acousticness,
            'valence': valence,
            'genre': genre_labels
        })
    
    @staticmethod
    def generate_image_colors(n_samples=1000, random_state=42):
        """Generate color distribution data for GMM"""
        np.random.seed(random_state)
        
        # Simulate image colors with overlapping Gaussians
        colors = []
        
        # Sky (blue)
        sky = np.random.multivariate_normal([100, 150, 220], 
                                            [[200, 0, 0], [0, 100, 0], [0, 0, 50]], 300)
        
        # Grass (green)
        grass = np.random.multivariate_normal([50, 150, 50],
                                              [[100, 0, 0], [0, 150, 0], [0, 0, 100]], 300)
        
        # Skin tone
        skin = np.random.multivariate_normal([200, 150, 120],
                                             [[150, 0, 0], [0, 100, 0], [0, 0, 80]], 200)
        
        # Shadow (dark)
        shadow = np.random.multivariate_normal([40, 40, 50],
                                               [[100, 0, 0], [0, 100, 0], [0, 0, 100]], 200)
        
        X = np.vstack([sky, grass, skin, shadow])
        X = np.clip(X, 0, 255)
        
        return pd.DataFrame({
            'red': X[:, 0],
            'green': X[:, 1],
            'blue': X[:, 2]
        })
    
    @staticmethod
    def generate_iot_sensor_data(n_samples=1000, random_state=42):
        """Generate IoT sensor data for OPTICS"""
        np.random.seed(random_state)
        
        # Multiple density clusters
        dense1 = np.random.multivariate_normal([10, 10], [[1, 0], [0, 1]], 300)
        dense2 = np.random.multivariate_normal([30, 10], [[0.5, 0], [0, 0.5]], 200)
        sparse = np.random.multivariate_normal([20, 30], [[5, 0], [0, 5]], 150)
        
        # Outliers
        outliers = np.random.uniform([0, 0], [40, 40], (50, 2))
        
        # Nested cluster
        nested_outer = np.random.multivariate_normal([35, 35], [[3, 0], [0, 3]], 200)
        nested_inner = np.random.multivariate_normal([35, 35], [[0.5, 0], [0, 0.5]], 100)
        
        X = np.vstack([dense1, dense2, sparse, outliers, nested_outer, nested_inner])
        
        return pd.DataFrame({
            'temperature': X[:, 0] + 20,
            'humidity': X[:, 1] * 2 + 30
        })
    
    @staticmethod
    def generate_grid_world(size=5):
        """Generate grid world for Q-Learning"""
        # 0: empty, 1: obstacle, 2: goal, 3: start
        grid = np.zeros((size, size))
        
        # Add obstacles
        grid[1, 1] = 1
        grid[2, 3] = 1
        grid[3, 1] = 1
        
        # Start and goal
        grid[0, 0] = 3  # Start
        grid[size-1, size-1] = 2  # Goal
        
        return grid
    
    @staticmethod
    def generate_simple_rl_env():
        """Generate simple environment info for RL demos"""
        return {
            'states': 25,
            'actions': 4,
            'actions_names': ['Up', 'Down', 'Left', 'Right'],
            'rewards': {
                'goal': 100,
                'step': -1,
                'obstacle': -10
            }
        }


def load_user_dataset(uploaded_file, target_column=None):
    """Load user-uploaded dataset"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel file.")
        
        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")


def prepare_features_target(df, target_column, feature_columns=None):
    """Prepare features and target from dataframe"""
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X, y


# ============================================
# DATASET ANALYSIS & AUTO-DETECTION
# ============================================
def analyze_dataset(df):
    """
    Analyze a dataset and auto-detect column types, suggest target variable.
    
    Returns dict with:
    - column_types: dict mapping column names to detected types
    - numeric_cols: list of numeric column names
    - categorical_cols: list of categorical column names
    - target_suggestion: suggested target column with confidence
    - data_quality: dict with missing values, duplicates info
    - task_suggestion: 'classification' or 'regression' based on target
    """
    analysis = {
        "column_types": {},
        "numeric_cols": [],
        "categorical_cols": [],
        "datetime_cols": [],
        "target_suggestion": None,
        "target_confidence": 0,
        "task_suggestion": None,
        "data_quality": {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "missing_values": {},
            "missing_total": 0,
            "duplicate_rows": 0,
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }
    }
    
    # Analyze each column
    for col in df.columns:
        dtype = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            unique_ratio = df[col].nunique() / len(df)
            
            # Check if it might be a binary/categorical encoded as numeric
            if df[col].nunique() <= 2 and df[col].nunique() > 0:
                analysis["column_types"][col] = "binary"
                analysis["categorical_cols"].append(col)
            elif df[col].nunique() <= 10 and unique_ratio < 0.05:
                analysis["column_types"][col] = "categorical_numeric"
                analysis["categorical_cols"].append(col)
            else:
                analysis["column_types"][col] = "numeric"
                analysis["numeric_cols"].append(col)
                
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            analysis["column_types"][col] = "datetime"
            analysis["datetime_cols"].append(col)
            
        else:
            analysis["column_types"][col] = "categorical"
            analysis["categorical_cols"].append(col)
        
        # Check missing values
        missing = df[col].isnull().sum()
        if missing > 0:
            analysis["data_quality"]["missing_values"][col] = missing
            analysis["data_quality"]["missing_total"] += missing
    
    # Check duplicate rows
    analysis["data_quality"]["duplicate_rows"] = df.duplicated().sum()
    
    # Suggest target column
    target_keywords = ['target', 'label', 'class', 'y', 'output', 'result', 'price', 
                       'approved', 'churned', 'is_', 'has_', 'score', 'rating']
    
    best_target = None
    best_score = 0
    
    for col in df.columns:
        score = 0
        col_lower = col.lower()
        
        # Check for keyword matches
        for keyword in target_keywords:
            if keyword in col_lower:
                score += 3
                break
        
        # Last column often is target
        if col == df.columns[-1]:
            score += 2
        
        # Binary columns are often targets
        if df[col].nunique() == 2:
            score += 2
        
        # Columns with few unique values might be classification targets
        if df[col].nunique() <= 10 and df[col].nunique() > 1:
            score += 1
        
        if score > best_score:
            best_score = score
            best_target = col
    
    if best_target:
        analysis["target_suggestion"] = best_target
        analysis["target_confidence"] = min(best_score / 7, 1.0)  # Normalize to 0-1
        
        # Suggest task type based on target
        target_unique = df[best_target].nunique()
        if target_unique <= 10:
            analysis["task_suggestion"] = "classification"
        else:
            analysis["task_suggestion"] = "regression"
    
    return analysis


def get_sample_datasets_info():
    """Get info about all available sample datasets"""
    return SAMPLE_DATASETS


def get_sample_dataset(dataset_key, n_samples=None):
    """
    Generate and return a sample dataset by key.
    
    Args:
        dataset_key: Key from SAMPLE_DATASETS
        n_samples: Optional override for number of samples
        
    Returns:
        DataFrame with the generated dataset
    """
    if dataset_key not in SAMPLE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(SAMPLE_DATASETS.keys())}")
    
    info = SAMPLE_DATASETS[dataset_key]
    generator_name = info["generator"]
    samples = n_samples or info["samples"]
    
    # Get the generator method from DatasetGenerator
    generator = getattr(DatasetGenerator, generator_name, None)
    
    if generator is None:
        raise ValueError(f"Generator not found: {generator_name}")
    
    # Call generator with appropriate args
    if dataset_key == "fraud_detection":
        return generator(n_samples=samples)
    elif dataset_key == "customer_segments":
        return generator(n_samples=samples)
    else:
        return generator(n_samples=samples)


def get_dataset_as_csv(dataset_key, n_samples=None):
    """Generate a sample dataset and return as CSV string"""
    df = get_sample_dataset(dataset_key, n_samples)
    return df.to_csv(index=False)

