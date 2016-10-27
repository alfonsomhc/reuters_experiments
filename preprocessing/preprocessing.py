"""
preprocessing.py

Feature preprocessing and extraction for text classification models in 
Reuters dataset.

Some sections of the code (e.g. tokenizer)are based on:
https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data
"""
from __future__ import print_function
import re
import os
from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.externals import joblib
from keras.preprocessing.text import Tokenizer as KerasTokenizer
from keras.preprocessing import sequence
cachedStopWords = stopwords.words("english")


def tokenize(text):
    """
    Tokenize an input text: lower case, remove stop words, stemming,
    discard words with characters other than letters
    """
    # Filter tokens
    min_length = 3
    # Lowercase
    words = map(lambda word: word.lower(), word_tokenize(text));
    # Remove stopwords
    words = [word for word in words if word not in cachedStopWords]
    # Stemming
    tokens =(list(map(lambda token: PorterStemmer().stem(token),
                  words)));
    # Remove words that not only contains words, and impose minimum length
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token: p.match(token)
        and len(token) >= min_length, tokens))
    return [ft.encode('utf8') for ft in filtered_tokens]


def create_dataset(raw_text_processor, max_words, **kwargs):
    """
    Compute data and targets for training and testing.
    Input:
        raw_text_processor: one of the processors defined in this file (below)
        max_words: maximum number of words to consider
        kwargs: key word arguments for the processor
    Output:
        data and targets for training and testing
        file_name: name of the pickle file used to cache the data
    """
    file_name = ("data/raw_text_processor_" + raw_text_processor +
                "_max_words_" + str(max_words) + "_" +
                str(kwargs).translate(None, """ '"{}""")
                .replace(":","_").replace(",","_") + ".pkl")

    if os.path.isfile(file_name):
        print('Read previously computed data and targets...')
        (train_data, train_targets,
            test_data, test_targets) = joblib.load(file_name)
    else:
        print('Create data and targets...')
        fileids = reuters.fileids()

        # Create data
        train_docs = [reuters.raw(doc_id) for doc_id in fileids
                        if doc_id.startswith("train")]
        test_docs = [reuters.raw(doc_id) for doc_id in fileids
                        if doc_id.startswith("test")]
        if raw_text_processor == "vector":
            train_data, test_data = raw_text_to_vector(
                train_docs, test_docs, max_words, kwargs["vectorizer"])
        elif raw_text_processor == "sequence":
            train_data, test_data = raw_text_to_sequences(
                train_docs, test_docs, max_words, kwargs["max_len"])

        # Create targets
        train_cats = [reuters.categories(doc_id) for doc_id in fileids
                        if doc_id.startswith("train")]
        test_cats = [reuters.categories(doc_id) for doc_id in fileids
                        if doc_id.startswith("test")]
        mlb = MultiLabelBinarizer()
        train_targets = mlb.fit_transform(train_cats)
        test_targets = mlb.transform(test_cats)
        
        # Cache data and targets
        joblib.dump((train_data, train_targets, test_data, test_targets),
            file_name)

    return (train_data, train_targets), (test_data, test_targets), file_name


def raw_text_to_sequences(train_docs, test_docs, max_words, max_len):
    """
    Transform raw text to a sequence (list of word indices)
    Tokenize with the function defined in this file, i.e. tokenize()
    KerasTokenizer is used to transform list of words to list of indices
    This involves creating a dummy text with the list of words.
    """
    t = KerasTokenizer(nb_words=max_words)
    t.fit_on_texts([' '.join(tokenize(doc)) for doc in train_docs])
    train_data = t.texts_to_sequences(
        [' '.join(tokenize(doc)) for doc in train_docs])
    test_data = t.texts_to_sequences(
        [' '.join(tokenize(doc)) for doc in test_docs])
    train_data = sequence.pad_sequences(train_data, maxlen=max_len)
    test_data = sequence.pad_sequences(test_data, maxlen=max_len)
    
    return train_data, test_data


def raw_text_to_vector(train_docs, test_docs, max_words, vectorizer):
    """
    Transform raw text to a vector, according to the bag of words model.
    We tokenize with the function defined in this file, i.e. tokenize()
    Scikit-learn is used to vectorize, using either counts or tfidf.
    """
    if vectorizer == 'count':
        vectorizer = CountVectorizer(binary=True, tokenizer=tokenize,
                        max_features=max_words, max_df=0.90);
    elif vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=max_words,
                        min_df=3, max_df=0.90, use_idf=True, sublinear_tf=True,
                        norm='l2');
    train_data = vectorizer.fit_transform(train_docs)
    test_data = vectorizer.transform(test_docs)

    return train_data, test_data