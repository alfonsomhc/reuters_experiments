from preprocessing.preprocessing import create_dataset, raw_text_to_vector
from mlp import train_evaluate_mlp

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(raw_text_to_vector, vectorizer = 'count', max_words = 1000)
train_evaluate_mlp(X_train, Y_train, X_test,Y_test, data_file = data_file, nb_hidden = 512, drop_out = 0.5)

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(raw_text_to_vector, vectorizer = 'count', max_words = 5000)
train_evaluate_mlp(X_train, Y_train, X_test, Y_test, data_file = data_file, nb_hidden = 512, drop_out = 0.5)

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(raw_text_to_vector, vectorizer = 'count', max_words = 7000)
train_evaluate_mlp(X_train, Y_train, X_test,Y_test, data_file = data_file, nb_hidden = 512, drop_out = 0.5)

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(raw_text_to_vector, vectorizer = 'count', max_words = 7000)
train_evaluate_mlp(X_train, Y_train, X_test,Y_test, data_file = data_file, nb_hidden = 512, drop_out = 0.2)

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(raw_text_to_vector, vectorizer = 'count', max_words = 7000)
train_evaluate_mlp(X_train, Y_train, X_test,Y_test, data_file = data_file, nb_hidden = 512, drop_out = 0.1)

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(raw_text_to_vector, vectorizer = 'count', max_words = 7000)
train_evaluate_mlp(X_train, Y_train, X_test, Y_test, data_file = data_file, nb_hidden = 200, drop_out = 0.5)

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(raw_text_to_vector, vectorizer = 'tfidf', max_words = 1000)
train_evaluate_mlp(X_train, Y_train, X_test, Y_test, data_file = data_file, nb_hidden = 512, drop_out = 0.5)

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(raw_text_to_vector, vectorizer = 'tfidf', max_words = 5000)
train_evaluate_mlp(X_train, Y_train, X_test, Y_test, data_file = data_file, nb_hidden = 512, drop_out = 0.5)

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(raw_text_to_vector, vectorizer = 'tfidf', max_words = 7000)
train_evaluate_mlp(X_train, Y_train, X_test, Y_test, data_file = data_file, nb_hidden = 512, drop_out = 0.5)

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(raw_text_to_vector, vectorizer = 'tfidf', max_words = 7000)
train_evaluate_mlp(X_train, Y_train, X_test, Y_test, data_file = data_file, nb_hidden = 200, drop_out = 0.5)

(X_train, Y_train),(X_test,Y_test), data_file = create_dataset(raw_text_to_vector, vectorizer = 'tfidf', max_words = 7000)
train_evaluate_mlp(X_train, Y_train, X_test, Y_test, data_file = data_file, nb_hidden = 512, drop_out = 0.2)