"""
tokenize.py

Simple tokenizer based on NLTK
"""

import re
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")

def tokenize(text):
    """
    Tokenize an input text: lower case, remove stop words, stemming, discard words with characters other than letters
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