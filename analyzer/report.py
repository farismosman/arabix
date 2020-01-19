import time, os
from utils import helpers
from clustering import kmeans
from processing import analysis, visualization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


current_dir = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    tweets = helpers.load_csv('%s/%s.csv'%(current_dir, time.strftime("%Y%m%d")))
    tweets = tweets[tweets['text'].notna()]

    term_frequencies = analysis.term_frequency(tweets['text'])
    term_frequencies = analysis.idf(tweets['text'], term_frequencies)
    term_frequencies.to_csv('%s/tf-%s.csv'%(current_dir, time.strftime("%Y%m%d")))

    visualization.histogram(term_frequencies, top=20)
    visualization.timeseries(term_frequencies, tweets, top=10, window='6H')

    tfidf_matrix, tfidf_features = analysis.tfidf(tweets['text'], vocabulary=None)
    tf_matrix, tf_features = analysis.tf(tweets['text'], vocabulary=None)

    tfidf = pd.DataFrame(data=tfidf_matrix, columns=tfidf_features)
    tf = pd.DataFrame(data=tf_matrix, columns=tf_features)

    tfidf = helpers.remove_features(tfidf, minscore=0.2, axis=0)
    tfidf = helpers.remove_features(tfidf, minscore=0.2, axis=1)
    tfidf = helpers.remove_features(tfidf, min_termlength=5)

    tf = helpers.remove_features(tf, minscore=100, axis=0)
    tf = helpers.remove_features(tf, minscore=5, axis=1)
    tf = helpers.remove_features(tf, min_termlength=5)

    features_df = kmeans.clusters(df=tfidf, ncluster=10)