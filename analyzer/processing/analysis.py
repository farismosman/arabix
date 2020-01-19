
import numpy as np
import pandas as pd
from utils import helpers

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from itertools import combinations
from collections import Counter



def term_frequency(df, word_column='word', frequency_column='frequency'):
    term_frequencies = df.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
    term_frequencies.columns = [word_column,frequency_column]
    return term_frequencies.groupby(word_column).sum().reset_index()


def idf(df, frequencies, word_column='word', idf_column='idf'):
    no_rows = df.shape[0]
    for i, word in enumerate(frequencies[word_column]):
        frequencies.loc[i, idf_column] = np.log(no_rows/(len(df[df.str.contains(word)])))
    return frequencies


def tfidf(data, vocabulary, no_features=5000):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words=helpers.stopwords_list, max_features=no_features, ngram_range=(1, 1), vocabulary=vocabulary)
    vectors = tfidf_vectorizer.fit_transform(data)
    return vectors.todense(), tfidf_vectorizer.get_feature_names()


def tf(data, vocabulary, no_features=5000):
    count_vectorizer = CountVectorizer(analyzer = "word", stop_words=helpers.stopwords_list, max_features=no_features, ngram_range=(1,1), vocabulary=vocabulary)
    vectors = count_vectorizer.fit_transform(data)
    return vectors.todense(), count_vectorizer.get_feature_names()


def co_occurrence_matrix(data, vocabulary, order=2, no_features=1000):
    count_vectorizer = CountVectorizer(analyzer = "word", stop_words=helpers.stopwords_list, max_features=no_features, ngram_range=(1, order), vocabulary=vocabulary)
    vectors = count_vectorizer.fit_transform(data)
    features = count_vectorizer.get_feature_names()

    df = pd.DataFrame(data=vectors.todense(), columns=features, index=data)
    return df