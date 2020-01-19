import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arabic_reshaper
from bidi import algorithm as bidialg



def frequencies(df, termcolumn, freqcolumn, min_termlength, top):
    _tf = df[df[termcolumn].str.len() > min_termlength]
    return _tf.sort_values(freqcolumn, ascending=False).head(top)


def histogram(df, min_termlength=5, termcolumn='word', freqcolumn='frequency', top=10, figsize=(15,15)):
    _df = frequencies(df, termcolumn, freqcolumn, min_termlength, top)

    terms = _df[termcolumn]
    _frequencies = _df[freqcolumn]
    indices = np.arange(len(terms))

    terms = [bidialg.get_display(arabic_reshaper.reshape(term)) for term in terms]
    plt.figure(figsize=figsize)
    plt.bar(indices, _frequencies)
    plt.xticks(indices, terms, rotation=45)
    plt.ylabel('Term Frequency')
    plt.title('Top ' + str(top) + ' Terms Frequency')


def timeseries(tf, tweets, termcolumn='word',
                  freqcolumn='frequency', min_termlength=5, top=10,
                  tweetcolumn='text', timecolumn='created_at', window='1H',
                  figsize=(15,15)):

    _tf = frequencies(tf, termcolumn, freqcolumn, min_termlength, top)

    result  = []
    terms = _tf[termcolumn]
    timestamp = 'timestamp'

    for term in terms:
        df = tweets[tweets[tweetcolumn].str.contains(term)]
        df = df[[timecolumn, tweetcolumn]]
        dates = pd.to_datetime(df[timecolumn])
        df = df.assign(timestamp=dates)
        df[freqcolumn] = np.ones(df.shape[0])
        df = df[[timestamp, freqcolumn]]
        df = df.groupby(pd.Grouper(key=timestamp, freq=window)).sum().reset_index()

        term = bidialg.get_display(arabic_reshaper.reshape(term))
        df[termcolumn] = term
        result.append(df)

    df = pd.concat(result)
    df = df.set_index(timestamp)
    df.index = pd.to_datetime(df.index)

    df.set_index(termcolumn, append=True)[freqcolumn].unstack().plot.bar(stacked=False, figsize=figsize)
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    ax = plt.axes()
    ax.xaxis.label.set_visible(False)