import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.data import load
from nltk.stem import SnowballStemmer
from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelWithLMHead

from textwrap import wrap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

class Twit_sentiment(Dataset):

    def __init__(self, twit, polarity, tokenizer, max_len):
        self.twit = twit
        self.polarity = polarity
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.twit)

    def __getitem__(self, item):
        twit = str(self.twit[item])

        encoding = tokenizer.encode_plus(twit,
                                         max_length=self.max_len,
                                         add_special_tokens=True,
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         return_token_type_ids=False,
                                         return_tensors='pt',
                                         truncation=True
                                         )
        return {
            'twit_content': twit,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(self.polarity[item], dtype=torch.long)
        }


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #         pooled_output = outputs[1]
        output = self.drop(pooled_output)
        #         output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)


def to_number(polarity):
    if polarity == 'NEU':
        return 0
    elif polarity == 'N':
        return 1
    else:
        return 2

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = Twit_sentiment(twit=df.content.to_numpy(),
                        polarity=df.polarity.to_numpy(),
                        tokenizer=tokenizer,
                        max_len=max_len
                       )
    return DataLoader(ds,
                      batch_size=batch_size,
                      num_workers= 4
                     )


def train_epoch(model,
                data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        target = d['targets'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, target)

        correct_predictions += torch.sum(preds == target)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model,
               data_loader,
               loss_fn,
               device,
               n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            target = d['targets'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, target)

            correct_predictions += torch.sum(preds == target)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def get_twits_predictions(model, data_loader):
    model = model.eval()

    twit_text = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d['twit_content']
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            target = d['targets'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)

            twit_text.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(target)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return twit_text, predictions, prediction_probs, real_values


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment');

