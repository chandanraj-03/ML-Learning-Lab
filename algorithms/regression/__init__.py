# Regression Algorithms
from .linear_regression import LinearRegressionDemo
from .lasso_regression import LassoRegressionDemo
from .ridge_regression import RidgeRegressionDemo
from .elastic_net import ElasticNetDemo
from .svr import SVRDemo
from .polynomial_regression import PolynomialRegressionDemo

__all__ = [
    'LinearRegressionDemo',
    'LassoRegressionDemo', 
    'RidgeRegressionDemo',
    'ElasticNetDemo',
    'SVRDemo',
    'PolynomialRegressionDemo'
]
