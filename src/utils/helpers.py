import pandas as pd

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords as sw


def stopwords(language):
    return sw.words(language)


def load_json(filename):
    return pd.read_json(filename, lines=True, encoding='utf-8')


def load_csv(filename):
    return pd.read_csv(filename)


def remove_features(df, minscore=0.01, axis=0, min_termlength=None):
    _df = df.copy()

    if min_termlength:
        features = _df.columns
        _features = features[features.str.len() > min_termlength]
        return _df[_features]

    _sum = _df.sum(axis=axis)
    _to_remove = _sum[_sum < minscore].index.values
    return _df.drop(columns=_to_remove) if axis == 0 else _df.drop(index=_to_remove)
