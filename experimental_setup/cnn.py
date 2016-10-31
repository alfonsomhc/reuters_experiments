"""
cnn.py

Parameterized experiment for Convolutional Neural Networks.
Some sections of the code are based on examples available in Keras library.
"""
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, merge
from keras.layers import Embedding, Convolution1D, GlobalMaxPooling1D
from evaluation.evaluate import optimize_threshold_for_fscore
import os

drop_out = 0.2
nb_epoch = 20
batch_size = 32

def train_evaluate_cnn(X_train, Y_train, X_test, Y_test, data_file, nb_hidden,
            max_words, max_len, embedding_dims, nb_filter, filter_length):
    """
    Document classification using a Convolutional Neural Network.
    Train a CNN for multi-label classification, then evaluate.

    Input:
        training and test data and targets
        data_file: file name for data cached in preprocessing module
        nb_hidden: number of hidden nodes
        max_words: number of words in the vocabulary
        max_len: number of words per document
        embedding_dims: dimension of embedding vectors
        nb_filter: number of feature extraction filters in convolutional layer
        filter_length: region size of the filters in convolutional layer
    """
    file_name = ("models/cnn_nb_hidden_" + str(nb_hidden) +
                    "_max_words_" + str(max_words) + "_" +
                    "_max_len_" + str(max_len) + "_" +
                    "_embedding_dims_" + str(embedding_dims) + "_" +
                    "_nb_filter_" + '_'.join(str(f) for f in nb_filter) + "_" +
                    "_filter_length_" + '_'.join(str(f) for f in filter_length)
                    + "_" + data_file.replace("data/", "").replace(".pkl", ".h5"))
    print(file_name)

    if os.path.isfile(file_name):
        print('Read previously trained model...')
        model = load_model(file_name)
    else:
        print('Building model...')

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        inputs = Input(shape=(max_len,), dtype='int32')
        embedding = Embedding(max_words,
                            embedding_dims,
                            input_length=max_len,
                            dropout=drop_out)(inputs)

        conv_layer = []
        
        for l,f in zip(filter_length, nb_filter):
            # we add a Convolution1D, which will learn nb_filter
            # word group filters of size filter_length:
            conv = Convolution1D(nb_filter=f,
                                    filter_length=l,
                                    border_mode='valid',
                                    activation='relu',
                                    subsample_length=1)(embedding)
            # we use max pooling:
            conv_layer.append(GlobalMaxPooling1D()(conv))

        conv_layer = merge(conv_layer, mode='concat')
        
        # We add a vanilla hidden layer:
        d1 = Dense(nb_hidden, activation = "relu")(conv_layer)
        d1_d = Dropout(drop_out)(d1)
        prediction = Dense(Y_train.shape[1], activation = "sigmoid")(d1_d)

        model = Model(input = inputs, output = prediction)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam')

        print('Training model...')
        history = model.fit(X_train, Y_train, nb_epoch=nb_epoch,
                            batch_size=batch_size, verbose=1, validation_split=0.1)
        # Cache model
        model.save(file_name)

    print('Evaluating model...')
    # For simplicity optimize threshold on test set
    # (should be done on validation test)
    fscore, threshold = optimize_threshold_for_fscore(model, X_test, Y_test)
    print("Best F-score = {}".format(fscore))
    print("Best threshold = {}".format(threshold))

    return fscore
