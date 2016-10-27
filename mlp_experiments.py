"""
mlp_experiments.py
"""
from preprocessing.preprocessing import create_dataset
from experimental_setup.mlp import train_evaluate_mlp
import pandas as pd
from utils.print_full_dataframe import print_full

results = []
for nb_hidden in [200, 512]:
    for drop_out in [0.1, 0.2, 0.5]:
        for max_words in [1000, 5000, 7000, 15000]:
            for vectorizer in ["count", "tfidf"]:
                (X_train, Y_train),(X_test, Y_test), data_file = create_dataset(
                    raw_text_processor = "vector",
                    max_words = max_words,
                    vectorizer = vectorizer)
                score = train_evaluate_mlp(X_train, Y_train, X_test, Y_test,
                    data_file = data_file,
                    nb_hidden = nb_hidden,
                    drop_out = drop_out)
                results.append(dict(nb_hidden=nb_hidden, vectorizer=vectorizer,
                    max_words=max_words,drop_out=drop_out, score=score))

results = pd.DataFrame(results)
print_full(results)
print(results.loc[results.score.idxmax,:])