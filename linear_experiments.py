"""
linear_experiments.py
"""
from preprocessing.preprocessing import create_dataset, raw_text_to_vector
from experimental_setup.linear import train_evaluate_linear
import pandas as pd
from utils.print_full_dataframe import print_full

results = []
for loss in ["log", "hinge"]:
    for vectorizer in ["count", "tfidf"]:
        for max_words in [1000, 5000, 7000, 15000]:
            for class_weight in [1,3,6,9,12]:
                (X_train, Y_train),(X_test,Y_test), data_file = create_dataset(
                    raw_text_processor = "vector",
                    max_words = max_words,
                    vectorizer = vectorizer)
                score = train_evaluate_linear(X_train, Y_train, X_test, Y_test,
                    data_file,
                    loss = loss,
                    class_weight = class_weight)
                results.append(dict(loss=loss, vectorizer=vectorizer,
                    max_words=max_words,class_weight=class_weight, score=score))

results = pd.DataFrame(results)
print_full(results)
print(results.loc[results.score.idxmax,:])
