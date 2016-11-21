"""
evaluate.py
"""
from sklearn import metrics
import numpy as np

def _optimize_threshold_for_metric(model, X_dev, Y_dev, metric):
    """
    Find the best score and threshold for a model and metric.
    Input:
        model: trained model with a predict method that outputs probabilities
            or scores (Keras' models, but not Scikit-Learn's).
        X_dev: development data
        Y_dev: development targets
        metric: scoring function
    Output:
        - best_score: optimal score
        - best_threshold: decision threshold that optimized score
    """
    best_score = 0
    best_threshold = 0
    for i in [0.05, 0.1, 0.3, 0.6]:
        try:
            X_dev = X_dev.todense()
        except AttributeError as e:
            if not "object has no attribute 'todense'" in e.message:
                raise
        preds = model.predict(X_dev)
        preds[preds>= i] = 1
        preds[preds< i] = 0
        score_i = metric(Y_dev, preds)
        if  score_i > best_score:
            best_score = score_i
            best_threshold = i
    return best_score, best_threshold
    
    
def optimize_threshold_for_fscore(model, X_dev, Y_dev):
    """
    Find the best F-score and threshold for a model.
    Input:
        model: trained model with a predict method that outputs probabilities
            or scores (Keras' models, but not Scikit-Learn's).
        X_dev: development data
        Y_dev: development targets
    Output:
        - best_fscore: optimal f-score
        - best_threshold: decision threshold that optimized f-score
    """
    
    metric = lambda y, pred: metrics.f1_score(y, pred, average ='macro')
    return _optimize_threshold_for_metric(model, X_dev, Y_dev, metric)


def optimize_threshold_for_fscore_det(model, X_dev, Y_dev):
    """
    Find the best F-score and threshold for a model (with single binary pred).
    Input:
        model: trained model with a predict method that outputs probabilities
            or scores (Keras' models, but not Scikit-Learn's).
        X_dev: development data
        Y_dev: development targets
    Output:
        - best_fscore: optimal f-score
        - best_threshold: decision threshold that optimized f-score
    """
    
    metric = lambda y, pred: metrics.f1_score(y, pred, average ='binary')
    return _optimize_threshold_for_metric(model, X_dev, Y_dev, metric)     