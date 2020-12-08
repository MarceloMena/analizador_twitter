from app import app
from flask import render_template, request, redirect, jsonify, make_response
from datetime import datetime

import tweepy
import webbrowser
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import torch
import nltk
import json

from PIL import Image
from transformers import BertTokenizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from app import twitter_sentanal
from app import twit_retriver
from app import remove_emoji
from app import TwitSentiment


@app.route("/", methods=["POST", "GET"])
def index():

    print(app.config["CONSUMER_KEY"])
    return render_template("main_page.html")


@app.route("/data", methods=["POST", "GET"])
def data():

    req = request.get_json()
    search_word = req["twit"]

    print('busqueda:', search_word)

    vect_sent = []
    if request.method == "POST":
        print(search_word)
        if True:
            retriever = twit_retriver.TwitRetriever()
            twits_df = retriever.twit_search(search_word=search_word)
            analizer = TwitSentiment.TwitSentiment()
            prediction = analizer.tokenize_twits(twits_df=twits_df)
            vect_sent = TwitSentiment.sent_discriminator(prediction)
            print(vect_sent)
            positive_words = TwitSentiment.word_separator(prediction, 'P')
            negative_words = TwitSentiment.word_separator(prediction, 'N')
    sent_vect = json.dumps(([vect_sent, positive_words]))
    res = make_response(sent_vect, 200)
    print(sent_vect)
    return res

@app.route("/funcionamiento")
def funcion():
    """
    docstring
    """
    return render_template("funcion.html")