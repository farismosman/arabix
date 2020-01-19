import time, os
import numpy as np
import pandas as pd
from utils import helpers
from utils.spellchecker import SpellChecker
from utils.tokenizer import Tokenizer

from nltk.stem.isri import ISRIStemmer


from alphabet_detector import AlphabetDetector
import preprocessor as tweet_processor
import re, json


current_dir = os.path.abspath(os.path.dirname(__file__))

stemmer = ISRIStemmer()

re_emoji = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U00010000-\U0010ffff" 
                    "]+", flags=re.UNICODE)

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                        """, re.VERBOSE)



def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def replace_chars(tweet):
    return tweet.replace('ة', 'ه')

def keep_only_arabic(words):
    ad = AlphabetDetector()
    tokens = [token for token in words if ad.is_arabic(token)]
    tweet = ''.join(tokens)
    return tweet

def remove_stopwords(tweet):
    for stopword in helpers.stopwords_list:
        tweet = tweet.replace(stopword, '')
    return tweet
    
def remove_emoji(tweet):
    return re_emoji.sub(r'', tweet)

def ar_stem(tweet):
    return stemmer.stem(tweet)


def remove_duplicate_chars(tweet):
    return re.sub(r'(.)\1+', r'\1', tweet)

def user_ids(users):
    ids = []
    for user in users:
        ids.append(user['id'])
    return ids

def remove_special_characters(tweet):
    return ''.join(re.sub(r'[^\w]+|_', ' ', tweet, flags=re.U))


def clean_tweet(tweet):
    tweet_processor.set_options(tweet_processor.OPT.URL,
                                tweet_processor.OPT.MENTION,
                                tweet_processor.OPT.HASHTAG,
                                tweet_processor.OPT.RESERVED,
                                tweet_processor.OPT.NUMBER
                                )
    tweet = tweet.lower()
    tweet = remove_emoji(tweet)
    tweet = remove_special_characters(tweet)
    tweet = keep_only_arabic(tweet)
    tweet = remove_stopwords(tweet)
    tweet = remove_diacritics(tweet)
    tweet = replace_chars(tweet)
    tweet = ar_stem(tweet)
    tweet = tweet_processor.clean(tweet)
    return tweet


if __name__ == "__main__":
    tweets = helpers.load_json('%s/data/tweets-%s.json'%(current_dir, '20190713'))
    tweets = tweets[['created_at', 'favorite_count', 'id', 'retweet_count', 'text', 'truncated', 'user']]
    tweets = tweets[tweets.truncated == False]
    tweets['user'] = user_ids(tweets.user.values)
    tweets['text'] = tweets['text'].map(clean_tweet)

    with open('%s/dictionary/arabic.json'%(current_dir)) as f:
        tokens = json.load(f)
    tokenizer = Tokenizer(dictionary=tokens)

    tweets['text'] = tweets['text'].map(tokenizer.tokenize)

    spellchecker = SpellChecker(tweets.text)

    tweets['text'] = tweets['text'].map(spellchecker.suggest)

    tweets.to_csv('%s/data/cleaned_tweets-%s.csv'%(current_dir, '20190713'))