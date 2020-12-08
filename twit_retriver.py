import tweepy
import webbrowser
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import torch
import nltk
from app import app

from PIL import Image
from transformers import BertTokenizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from app import twitter_sentanal
from app import remove_emoji

consumer_key = app.config["CONSUMER_KEY"]
consumer_secret = app.config["CONSUMER_SECRET"]
access_token = app.config["ACCESS_TOKEN"]
access_token_secret = app.config["ACCESS_TOKEN_SECRET"]
callback_url = app.config["CALLBACK_URL"]
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# search_word = 'Trump'  # Texto de busqueda del navegador buscar en HTML #form-control
num_twits = 100
start_since = "2020-08-05"
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


class TwitRetriever:
    def __init__(self):
        self.num_twits = num_twits
        self.start_date = start_since
        self.api = api

    def twit_search(self, search_word):
        twits_df = []
        search_words = search_word + ' -filter:retweets'
        for tweet in tweepy.Cursor(self.api.search,
                                   q=search_words,
                                   lang="es",
                                   tweet_mode="extended",
                                   since=self.start_date).items(self.num_twits):
            no_url = re.sub(
                r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<
                >]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
                " ", tweet.full_text)
            no_jump = re.sub(r'\n', "", no_url)
            twits_df.append(no_jump)

        return twits_df
