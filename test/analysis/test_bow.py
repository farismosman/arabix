import pandas as pd
from unittest import TestCase
from src.analysis.bow import BoW


class TestBoW(TestCase):

  def setUp(self):
    self.df = pd.read_csv('test/data/tweets.csv')

  def test_tf_should_return_vectorized_documents_and_features(self):
    bow = BoW(
      data=self.df['text'],
      language='arabic',
      ngram=(1, 1),
      vocabulary=None,
      no_features=None)

    vectors, features = bow.tf()

    self.assertEqual(vectors.shape, (5751, 19147))
    self.assertEqual(len(features), 19147)

  def test_tf_should_return_vectorized_co_occurance_documents_and_features(self):
    bow = BoW(
      data=self.df['text'],
      language='arabic',
      ngram=(1, 2),
      vocabulary=None,
      no_features=None)

    vectors, features = bow.tf()

    self.assertEqual(vectors.shape, (5751, 52411))
    self.assertEqual(len(features), 52411)