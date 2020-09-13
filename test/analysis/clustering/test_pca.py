import pandas as pd
from unittest import TestCase
from src.analysis.bow import BoW
from src.analysis.clustering.pca import PCA


class TestPCA(TestCase):

  def setUp(self):
    df = pd.read_csv('test/data/tweets.csv')
    bow = BoW(
      data=df['text'],
      language='arabic',
      ngram=(1, 1),
      vocabulary=None,
      no_features=None)

    self.vectors, _ = bow.tf()

  def test_cluster_should_reduce_the_dimensions_of_word_vector(self):
    pca = PCA(bow=self.vectors, no_components=1)

    clusters = pca.cluster()

    self.assertEqual(len(clusters), 5751)
    self.assertEqual(len(clusters[0]), 1)

    pca = PCA(bow=self.vectors, no_components=2)

    clusters = pca.cluster()

    self.assertEqual(len(clusters), 5751)
    self.assertEqual(len(clusters[0]), 2)