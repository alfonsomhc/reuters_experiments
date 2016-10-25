"""
cnn_experiments.py
"""
from preprocessing.preprocessing import create_dataset
from param_experiment.cnn import train_evaluate_cnn

for nb_hidden in [200, 512]:
    for embedding_dims in [20, 50, 100, 200]:
        for max_words in [1000, 5000, 7000]:
            for max_len in [200, 500]:
                for nb_filter in [100, 200, 500]:
                    for filter_length in [3, 5, 8]:
                        (X_train, Y_train),(X_test,Y_test), data_file = create_dataset(
                            "sequence", 
                            max_words = max_words, 
                            max_len = max_len)
                        train_evaluate_cnn(X_train, Y_train, X_test, Y_test, data_file = data_file, 
                            nb_hidden = nb_hidden, 
                            max_words = max_words, 
                            max_len = max_len, 
                            embedding_dims = embedding_dims, 
                            nb_filter = nb_filter, 
                            filter_length = filter_length)
    
