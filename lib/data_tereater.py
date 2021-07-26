# from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.initializers import Constant
# from tensorflow.keras.models import Sequential
# from tqdm import tqdm
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# import string
# import gensim
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.util import ngrams
from spellchecker import SpellChecker
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import Counter
import itertools
import string


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


def cleaned_text(text):
    text = re.sub("http\S+", "", text)
    text = re.sub("pic.twitter\S+", "", text)
    text = re.sub("@\S+", "", text)
    text = re.sub('#', '', text)
    text = re.sub('goooooooaaaaaal', 'goal', text)
    text = re.sub('SOOOO', 'SO', text)
    text = re.sub('LOOOOOOL', 'LOL', text)
    text = re.sub('Cooool', 'cool', text)
    text = re.sub('|', '', text)
    text = re.sub(r'\?{2,}', '? ', text)
    text = re.sub(r'\.{2,}', '. ', text)
    text = re.sub(r'\!{2,}', '! ', text)
    text = re.sub('&amp;', '&', text)
    text = re.sub('Comin', 'Coming', text)
    text = re.sub('&gt;', '> ', text)
    text = re.sub('&lt;', '< ', text)
    text = re.sub(r'.:', '', text)
    text = re.sub('baaaack', 'back', text)
    text = re.sub('RT', '', text)
    text = re.sub('\s{2,}', ' ', text)
    text = text.lower()
    return text


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def correct_spellings(text):
    spell = SpellChecker()
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


df = pd.concat([train, test])
df['text'] = df['text'].apply(lambda x: remove_emoji(x))
df['text'] = df['text'].apply(lambda x: remove_html(x))
df['text'] = df['text'].apply(lambda x: remove_URL(x))
df['text'] = df['text'].apply(lambda x: remove_punct(x))
df['text'] = df['text'].apply(lambda x: correct_spellings(x))
