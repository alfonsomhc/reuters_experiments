"""
mlp.py

Parameterized experiment for Multi-Layer Perceptrons.
Some sections of the code are based on examples available in Keras library.
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from evaluation.evaluate import optimize_threshold_for_fscore
import os

batch_size = 32
nb_epoch = 5

def train_evaluate_mlp(X_train, Y_train, X_test, Y_test, data_file, nb_hidden, drop_out):
    """
    Document classification based on bag of words and Multi-Layer Perceptron.
    Train a MLP for multi-label classification, then evaluate.
    Input:
        training and test data and targets
        data_file: file name for data cached in preprocessing module
        nb_hidden: number of hidden nodes
        drop_out: drop out rate
    """
    file_name = ("models/mlp_nb_hidden_" + str(nb_hidden) +
                    "_drop_out_" + str(drop_out) + "_" +
                    data_file.replace("data/", "").replace(".pkl", ".h5"))
    print(file_name)

    if os.path.isfile(file_name):
        print('Read previously trained model...')
        model = load_model(file_name)
    else:
        print('Building model...')
        model = Sequential()
        model.add(Dense(nb_hidden, input_shape=(X_train.shape[1],)))
        model.add(Activation('relu'))
        model.add(Dropout(drop_out))
        model.add(Dense(Y_train.shape[1]))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')

        print('Training model...')
        history = model.fit(X_train.todense(), Y_train, nb_epoch=nb_epoch,
                            batch_size=batch_size, verbose=2, validation_split=0.1)
        # Cache model
        model.save(file_name)

    print('Evaluating model...')
    # For simplicity optimize threshold on test set
    # (should be done on validation test)
    fscore, threshold = optimize_threshold_for_fscore(model, X_test, Y_test)
    print("Best F-score = {}".format(fscore))
    print("Best threshold = {}".format(threshold))

    return fscore