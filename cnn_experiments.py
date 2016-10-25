from preprocessing.preprocessing import create_dataset, raw_text_to_sequences
from cnn import train_evaluate_cnn

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(
    raw_text_to_sequences, 
    max_words = 10000, 
    max_len = 500)
train_evaluate_cnn(X_train, Y_train, X_test, Y_test, data_file = data_file, 
    nb_hidden = 512, 
    max_words = 10000, 
    max_len = 500, 
    embedding_dims = 200, 
    nb_filter = 500, 
    filter_length = 5)