import pandas as pd
from unittest import TestCase
from src.analysis.bow import BoW


class TestBoW(TestCase):

  def setUp(self):
    self.df = pd.read_csv('test/data/tweets.csv')
    self.bow = BoW(data=self.df['text'], vocabulary=None, language='arabic')

  def test_tf_should_return_vectorized_documents_and_features(self):
    vectors, features = self.bow.tf(no_features=None)

    self.assertEqual(vectors.shape, (5751, 19147))
    self.assertEqual(len(features), 19147)

  def test_cooccurrence_should_return_vectorized_documents_and_features(self):
    vectors, features = self.bow.cooccurrence(no_features=None, order=2)

    self.assertEqual(vectors.shape, (5751, 52411))
    self.assertEqual(len(features), 52411)