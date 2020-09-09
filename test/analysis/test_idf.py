import pandas as pd
from unittest import TestCase
from src.analysis.idf import IDF


class TestIDF(TestCase):
  def setUp(self):
    self.df = pd.read_csv('test/data/tweets.csv')
    self.idf = IDF(
      data=self.df['text'],
      language='arabic',
      ngrams=(1, 1),
      no_features=None,
      vocabulary=None)

  def test_idf_should_return_inverse_doc_frequency_and_features(self):
    _idf, features = self.idf.idf()

    self.assertEqual(len(_idf), 19147)
    self.assertEqual(len(features), 19147)

  def test_tfidf_should_return_term_frequency_inverse_doc_frequency_and_features(self):
    _tfidf, features = self.idf.tfidf()

    self.assertEqual(len(_tfidf), 5751)
    self.assertEqual(len(features), 19147)

    