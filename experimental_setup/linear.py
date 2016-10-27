"""
linear.py

Parameterized experiment for linear models.
"""
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.externals import joblib

def train_evaluate_linear(X_train, Y_train, X_test, Y_test, data_file, loss, class_weight):
    """
    Document classification based on bag-of-words and linear models.
    Train a linear model for each detection class and evaluate the system.

    Input:
        training and test data and targets
        data_file: file name for data cached in preprocessing module
        loss: name of the loss function
        class_weight: weight for the detection class
    Output:
        Average f-score (macro average)
    """
    nb_classes = Y_train.shape[1]
    file_name = ("models/linear_loss_" + str(loss) +
                "_class_weight_" + str(class_weight) + "_" +
                data_file.replace("data/", ""))
    print(file_name)

    if os.path.isfile(file_name):
        print('Read previously trained model...')
        model = joblib.load(file_name)
    else:
        print('Training models...')
        model = {}
        for i in xrange(0, nb_classes):
            model[i] = SGDClassifier(loss = loss, class_weight = {1:class_weight})
            train_targets = Y_train[:,i]
            model[i].fit(X_train, train_targets)
        # Cache models
        joblib.dump(model, file_name)

    print('Evaluating models...')
    fscore = [None]* nb_classes
    for i in xrange(0, nb_classes):
        test_targets = Y_test[:,i]
        predicted = model[i].predict(X_test)
        fscore[i] = f1_score(test_targets, predicted)
    # Compute macro average for f-score
    average_fscore = sum(fscore)/len(fscore)
    print("Average F-Score = {}".format(average_fscore))

    return average_fscore