
from src.utils import helpers

from sklearn.feature_extraction.text import CountVectorizer

class BoW:
    def __init__(self, data, language, ngram, no_features, vocabulary):
        self.data = data
        self.vectorizer = self.__vectorizer__(
            language,
            ngram,
            no_features,
            vocabulary)

    def tf(self,):
        vectors = self.vectorizer.fit_transform(self.data)
        return vectors.todense(), self.vectorizer.get_feature_names()

    def __vectorizer__(self, language, ngram, no_features, vocabulary):
        vectorizer = CountVectorizer(
            analyzer = "word",
            stop_words=helpers.stopwords(language),
            max_features=no_features,
            ngram_range=ngram,
            vocabulary=vocabulary)

        vectorizer.fit(self.data)
        return vectorizer