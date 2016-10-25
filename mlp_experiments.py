"""
mlp_experiments.py
"""
from preprocessing.preprocessing import create_dataset
from param_experiment.mlp import train_evaluate_mlp

for nb_hidden in [200, 512]:
    for drop_out in [0.1, 0.2, 0.5]:
        for max_words in [1000, 5000, 7000]:
            for vectorizer in ["count", "tfidf"]:
                (X_train, Y_train),(X_test, Y_test), data_file = create_dataset("vector", 
                    max_words = max_words,
                    vectorizer = vectorizer)
                train_evaluate_mlp(X_train, Y_train, X_test, Y_test, data_file = data_file, 
                    nb_hidden = nb_hidden, 
                    drop_out = drop_out)