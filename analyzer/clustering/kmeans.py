import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import maxabs_scale

import arabic_reshaper
from bidi import algorithm as bidialg

from wordcloud import WordCloud


def clusters(df, ncluster=10):
    _df = maxabs_scale(df)
    model = KMeans(n_clusters=ncluster, precompute_distances="auto", n_jobs=-1)
    labels = model.fit_predict(_df)
    df['labels'] = labels

    return df


def word_cloud(df, top=25, figsize=(15,15), cloud_width=800, cloud_height=400, cloud_font='fonts/ANefelAdeti.ttf', cloud_label=1):
    def top_features(df, top):
        _features = df.mean(axis=0)
        _features = _features[_features > 0]
        if _features.shape[0] < top:
            _features = _features.to_frame().reset_index()
        else:
            _features = _features.head(top).to_frame().reset_index()
        _features.columns = ['feature', 'score']
        return _features

    df = top_features(df, top)
    wordcloud = WordCloud(font_path=cloud_font, mode='RGB', width=cloud_width, height=cloud_height)
    terms = [bidialg.get_display(arabic_reshaper.reshape(term)) for term in df['feature']]

    features = pd.Series(df['score'].values, index=terms).to_dict()
    wordcloud.generate_from_frequencies(frequencies=features)
    plt.figure(figsize=figsize, facecolor='k')
    plt.imshow(wordcloud)
    title = plt.title('Lable: ' + str(cloud_label))
    plt.getp(title)
    plt.getp(title, 'text')
    plt.setp(title, color='w')
    plt.axis("off")