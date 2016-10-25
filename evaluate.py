"""
evaluate.py
"""
from sklearn import metrics
import numpy as np

def optimize_threshold_for_fscore(model, X_dev, Y_dev):
    """
    """
    best_fscore = 0
    best_threshold = 0
    for i in np.arange(0,1,0.1):
        preds = model.predict(X_dev.todense())
        preds[preds>= i] = 1
        preds[preds< i] = 0
        fscore_i = metrics.f1_score(Y_dev, preds, average ='macro')
        if  fscore_i > best_fscore:
            best_fscore = fscore_i
            best_threshold = i
    return best_fscore, best_threshold 