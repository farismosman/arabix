
import numpy as np
from src.utils import helpers

from sklearn.feature_extraction.text import TfidfVectorizer

class IDF():
    def __init__(self, data, language, ngrams, no_features, vocabulary):
        self.data = data
        self.vectorizer = self.__vectorizer__(
            language,
            no_features,
            ngrams,
            vocabulary)

    def idf(self):
        return self.vectorizer.idf_, self.vectorizer.get_feature_names()

    def tfidf(self):
        vectors = self.vectorizer.transform(self.data) 
        return vectors.todense(), self.vectorizer.get_feature_names()

    def __vectorizer__(self, language, no_features, ngrams, vocabulary):
        vectorizer = TfidfVectorizer(
            analyzer='word',
            stop_words=helpers.stopwords(language),
            max_features=no_features,
            ngram_range=ngrams,
            use_idf=True,
            smooth_idf=True,
            vocabulary=vocabulary)

        vectorizer.fit(self.data)
        return vectorizer