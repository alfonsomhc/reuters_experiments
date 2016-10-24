"""
preprocessing.py

Simple tokenizer based on NLTK
"""

from __future__ import print_function
import re
from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

max_words = 1000

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
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens));
    return [ft.encode('utf8') for ft in filtered_tokens]

count_vect = CountVectorizer(binary=True, tokenizer=tokenize, max_df=0.90,
                                max_features=max_words);

tfiddf_vect = TfidfVectorizer(tokenizer=tokenize, min_df=3, max_df=0.90,
                    max_features=max_words, use_idf=True, sublinear_tf=True,
                    norm='l2');

def create_dataset(raw_text_processor, **kwargs):
    """
    Compute data and targets for training and testing sets (Reuters dataset)
    Input: vectorizer
    Output: tuples with data and targets
    """
    print('Create data and targets...')

    fileids = reuters.fileids()

    # Create data
    train_docs = [reuters.raw(doc_id) for doc_id in fileids
                    if doc_id.startswith("train")]
    test_docs = [reuters.raw(doc_id) for doc_id in fileids
                    if doc_id.startswith("test")]
    train_data, test_data = raw_text_processor(train_docs, test_docs, **kwargs)

    # Create targets
    train_cats = [reuters.categories(doc_id) for doc_id in fileids
                    if doc_id.startswith("train")]
    test_cats = [reuters.categories(doc_id) for doc_id in fileids
                    if doc_id.startswith("test")]
    mlb = MultiLabelBinarizer()
    train_targets = mlb.fit_transform(train_cats)
    test_targets = mlb.transform(test_cats)

    return (train_data, train_targets), (test_data, test_targets)


def raw_text_to_sequences(train_docs, test_docs):
    """
    max_features, maxlen
    """
    t = Tokenizer(nb_words=max_features)
    t.fit_on_texts([' '.join(tokenize(doc)) for doc in train_docs])
    train_data = t.texts_to_sequences([' '.join(tokenize(doc)) for doc in train_docs])
    test_data = t.texts_to_sequences([' '.join(tokenize(doc)) for doc in test_docs])
    train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
    test_data = sequence.pad_sequences(test_data, maxlen=maxlen)
    return train_data, test_data


def raw_text_to_vector(train_docs, test_docs, vectorizer = None):
    """

    """
    train_data = vectorizer.fit_transform(train_docs)
    test_data = vectorizer.transform(test_docs)
    return train_data, test_data