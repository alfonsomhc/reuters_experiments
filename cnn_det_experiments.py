"""
cnn_experiments.py
"""
from preprocessing.preprocessing import create_dataset
from experimental_setup.cnn_det import train_evaluate_cnn
import pandas as pd
from utils.print_full_dataframe import print_full
from itertools import izip, product

results = []
parameters = dict(
    raw_text_processor = ["sequence"],
    nb_hidden =[200], 
    embedding_dims = [100], 
    max_words = [15000], 
    max_len = [500],
    nb_filter = [[50, 50]],
    filter_length = [[3,8]],
    class_weight = [15]
)

param_iter = (dict(izip(parameters, x)) for x in product(*parameters.itervalues()))
    
for param in param_iter:
    (X_train, Y_train),(X_test,Y_test), data_file = create_dataset(**param)
    param_iter["score"] =  train_evaluate_cnn(X_train, Y_train, X_test, Y_test,
        data_file = data_file,
        **param)
    results.append(param_iter)

results = pd.DataFrame(results)
print_full(results)
print(results.loc[results.score.idxmax,:])

