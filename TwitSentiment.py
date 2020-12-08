import tweepy
import webbrowser
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import torch
import nltk

from PIL import Image
from transformers import BertTokenizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from app import twitter_sentanal
from app import twit_retriver
from app import remove_emoji

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAINED_MODEL_DIR = './app/best_model_state_0109_medium.bin'
PRE_TRAINED_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
class_name = ['NEU', 'N', 'P']

model = twitter_sentanal.SentimentClassifier(3)
model.load_state_dict(torch.load(TRAINED_MODEL_DIR))
model = model.to(device)


class TwitSentiment:
    def __init__(self):
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    def tokenize_twits(self, twits_df):
        sentiment = pd.DataFrame(columns=['twit_content', 'polarity'])
        for i, twit in enumerate(twits_df):
            encoded_twit = self.tokenizer.encode_plus(twit,
                                                      max_length=64,
                                                      add_special_tokens=True,
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True,
                                                      return_token_type_ids=False,
                                                      return_tensors='pt',
                                                      truncation=True)
            input_ids = encoded_twit['input_ids'].to(device)
            attention_mask = encoded_twit['attention_mask'].to(device)
            output = self.model(input_ids, attention_mask)
            _, prediction = torch.max(output, dim=1)
            sentiment.loc[i] = [twit, class_name[prediction]]
            sentiment.polarity[i] = class_name[prediction]

        return sentiment


def sent_discriminator(prediction_twits):
    sent_neg = prediction_twits.polarity.value_counts()['N']
    sent_pos = prediction_twits.polarity.value_counts()['P']
    sent_neu = prediction_twits.polarity.value_counts()['NEU']
    return [int(sent_neg), int(sent_pos), int(sent_neu)]


def graficator_matplot(vect_sent, search_subject):
    category_names = ['Negativo', 'Positivo', 'Neutro']
    sizes = vect_sent
    offset = [0.05, 0.05, 0.05]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes,
            labels=category_names,
            textprops={'fontsize': 15},
            startangle=90,
            autopct='%1.0f%%',
            pctdistance=0.8,
            explode=offset)

    center_circle = plt.Circle((0, 0), radius=0.6, fc='white')
    plt.gca().add_artist(center_circle)
    plt.title(f'Analisis de sentimientos en twiter\n para la busqueda: {search_subject}', fontsize=20)
    plt.show()


def word_separator(content, sentiment_tag):
    sentiment_content = content[content.polarity == sentiment_tag]
    word_list = [''.join(word) for word in sentiment_content.twit_content]
    content_as_string = ' '.join(word_list)
    no_emoji = remove_emoji.remove_emoji(content_as_string)
    text_tokens = word_tokenize(remove_emoji.remove_emoji(no_emoji))
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('spanish')]
    clean_text = ' '.join(tokens_without_sw)
    return clean_text


def word_image(clean_text, image_dir, color_map):
    icon = Image.open(image_dir)
    image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
    image_mask.paste(icon, box=icon)
    rgb_array = np.array(image_mask)

    word_cloud = WordCloud(mask=rgb_array, background_color='white',
                           max_words=100, colormap=color_map)
    word_cloud.generate(clean_text)

    plt.figure(figsize=[8, 8])
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')

    plt.show()


# search_word = input('Tema a buscar: ')
# retriever = twit_retriver.TwitRetriever()
# twits_df = retriever.twit_search(search_word=search_word)
# # print(twits_df)
# analizer = TwitSentiment()
# prediction = analizer.tokenize_twits(twits_df=twits_df)
# # print(prediction.polarity.value_counts())
# vect_sent = sent_discriminator(prediction)
# graficator_matplot(vect_sent=vect_sent, search_subject=search_word)
# positive_words = word_separator(prediction, 'P')
# negative_words = word_separator(prediction, 'N')
# word_image(clean_text=positive_words, image_dir='images/thumbs-up.png', color_map='summer')
# word_image(clean_text=negative_words, image_dir='images/thumb-down.png', color_map='autumn')
