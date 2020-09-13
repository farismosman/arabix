from sklearn.decomposition import PCA as SKLearnPCA


class PCA:
  def __init__(self, bow, no_components):
    self.bow = bow
    self.no_components = no_components
    self.pca = None

  def cluster(self):
    pca = SKLearnPCA(n_components=self.no_components)
    vectors = pca.fit_transform(self.bow)
    self.pca = pca
    return vectors