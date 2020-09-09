
from src.utils import helpers

from sklearn.feature_extraction.text import CountVectorizer

class BoW:
    def __init__(self, data, vocabulary, language):
        self.data = data
        self.vocabulary = vocabulary
        self.language = language

    def tf(self, no_features, ngram=(1, 1)):
        count_vectorizer = CountVectorizer(
            analyzer = "word",
            stop_words=helpers.stopwords(self.language),
            max_features=no_features,
            ngram_range=ngram,
            vocabulary=self.vocabulary)

        vectors = count_vectorizer.fit_transform(self.data)

        return vectors.todense(), count_vectorizer.get_feature_names()


    def cooccurrence(self, order, no_features):
        return self.tf(no_features=no_features, ngram=(1, order))