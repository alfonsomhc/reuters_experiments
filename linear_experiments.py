"""
linear_experiments.py
"""
from preprocessing.preprocessing import create_dataset, raw_text_to_vector
from param_experiment.linear import train_evaluate_linear


for loss in ["log", "hinge"]:
    for vectorizer in ["count", "tfidf"]:
        for max_words in [1000, 5000, 7000]:
            for class_weight in [1,3,6,9,12]:
                (X_train, Y_train),(X_test,Y_test), data_file = create_dataset("vector", 
                    max_words = max_words,
                    vectorizer = vectorizer)
                train_evaluate_linear(X_train, Y_train, X_test, Y_test, data_file, 
                    loss = loss, 
                    class_weight = class_weight)
    
