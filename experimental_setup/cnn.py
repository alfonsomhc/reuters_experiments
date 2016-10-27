"""
cnn.py
"""
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, Convolution1D, MaxPooling1D
from evaluation.evaluate import optimize_threshold_for_fscore
import os

drop_out = 0.2
nb_epoch = 20
batch_size = 32

def train_evaluate_cnn(X_train, Y_train, X_test, Y_test, data_file, nb_hidden,
            max_words, max_len, embedding_dims, nb_filter, filter_length):
    """
    """

    file_name = ("models/cnn_nb_hidden_" + str(nb_hidden) +
                    "_max_words_" + str(max_words) + "_" +
                    "_max_len_" + str(max_len) + "_" +
                    "_embedding_dims_" + str(embedding_dims) + "_" +
                    "_nb_filter_" + str(nb_filter) + "_" +
                    "_filter_length_" + str(filter_length) + "_" +
                    data_file.replace("data/", "").replace(".pkl", ".h5"))
    print(file_name)

    if os.path.isfile(file_name):
        print('Read previously trained model...')
        model = load_model(file_name)
    else:
        print('Building model...')
        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(max_words,
                            embedding_dims,
                            input_length=max_len,
                            dropout=drop_out))

        # we add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length:
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1))
        # we use max pooling:
        model.add(MaxPooling1D(pool_length=model.output_shape[1]))

        # We flatten the output of the conv layer,
        # so that we can add a vanilla dense layer:
        model.add(Flatten())

        # We add a vanilla hidden layer:
        model.add(Dense(nb_hidden))
        model.add(Dropout(drop_out))
        model.add(Activation('relu'))

        model.add(Dense(Y_train.shape[1]))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam')
        print('Training model...')
        history = model.fit(X_train, Y_train, nb_epoch=nb_epoch,
                            batch_size=batch_size, verbose=1, validation_split=0.1)
        model.save(file_name)

    print('Evaluating model...')
    # For simplicity optimize threshold on test set
    # (should be done on validation test)
    fscore, threshold = optimize_threshold_for_fscore(model, X_test, Y_test)
    print("Best F-score = {}".format(fscore))
    print("Best threshold = {}".format(threshold))

    return fscore