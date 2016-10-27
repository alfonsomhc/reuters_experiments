"""
cnn_experiments.py
"""
from preprocessing.preprocessing import create_dataset
from experimental_setup.cnn import train_evaluate_cnn
import pandas as pd
from utils.print_full_dataframe import print_full

results = []
for nb_hidden in [100]:
    for embedding_dims in [20]:
        for max_words in [15000]:
            for max_len in [500]:
                for nb_filter in [150]:
                    for filter_length in [12]:
                        (X_train, Y_train),(X_test,Y_test), data_file = create_dataset(
                            raw_text_processor = "sequence",
                            max_words = max_words,
                            max_len = max_len)
                        score =  train_evaluate_cnn(X_train, Y_train, X_test, Y_test,
                            data_file = data_file,
                            nb_hidden = nb_hidden,
                            max_words = max_words,
                            max_len = max_len,
                            embedding_dims = embedding_dims,
                            nb_filter = nb_filter,
                            filter_length = filter_length)
                        results.append(dict(
                            nb_hidden=nb_hidden, max_len=max_len,
                            max_words=max_words,
                            score=score, nb_filter=nb_filter,
                            filter_length=filter_length,
                            embedding_dims=embedding_dims))

results = pd.DataFrame(results)
print_full(results)
print(results.loc[results.score.idxmax,:])

