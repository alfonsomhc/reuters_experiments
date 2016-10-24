"""
mlp.py

Document classification based on Multilayer Perceptron and bag of words
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn import metrics

batch_size = 32
nb_epoch = 10

def train_evaluate_mlp(X_train, Y_train, X_test,Y_test):
    
    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_shape=(X_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Y_train.shape[1]))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    print('Training model...')
    history = model.fit(X_train.todense(), Y_train, nb_epoch=nb_epoch, 
                        batch_size=batch_size, verbose=1, validation_split=0.1)
    
    print('Evaluating model...')
    # For simplicity optimize threshold on test set 
    # (should be done on validation test)
    best_fscore = 0
    best_threshold = 0
    for i in np.arange(0,1,0.1):
        preds = model.predict(X_test.todense())
        preds[preds>= i] = 1
        preds[preds< i] = 0
        fscore_i = metrics.f1_score(Y_test, preds, average ='macro')
        print("F-score = {}".format(fscore_i))
        print("Threshold = {}".format(i))
        if  fscore_i > best_fscore:
            best_fscore = fscore_i
            best_threshold = i
        
    print("Best F-score = {}".format(best_fscore))
    print("Best threshold = {}".format(best_threshold))