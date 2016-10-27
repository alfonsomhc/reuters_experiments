## Text Classification Experiments on Reuters

Implementation of three classification models for the Reuters task:

1. Linear models
2. Multi Layer Perceptron
3. Convolutional Neural Network

### Folder structure
Folder description:
 - data: preprocessing module caches computed features in this folder
 - evaluation: F-score evaluation of MLP and CNN
 - experimental_setup: parameterized experiments
 - models: experiments cache trained models in this folder
 - preprocessing: feature preprocessing and extraction
 - results: experiment outputs are stored here
 - utils: general purpose useful functions

### Dependencies
Tested on Python Anaconda distribution.
Preprocessing, models and evaluation rely on Scikit-learn, Keras and Nltk.
