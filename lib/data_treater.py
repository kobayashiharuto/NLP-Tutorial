from spellchecker import SpellChecker
import re
import pandas as pd
import matplotlib.pyplot as plt
import string
from tqdm import tqdm


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()


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
    emoji_pattern = re.compile('['
                               u'\U0001F600-\U0001F64F'  # emoticons
                               u'\U0001F300-\U0001F5FF'  # symbols & pictographs
                               u'\U0001F680-\U0001F6FF'  # transport & map symbols
                               u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                               u'\U00002702-\U000027B0'
                               u'\U000024C2-\U0001F251'
                               ']+', flags=re.UNICODE)
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
    return ' '.join(corrected_text)


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

df = pd.concat([train, test])

df['text'] = df['text'].apply(lambda x: remove_emoji(x))
df['text'] = df['text'].apply(lambda x: remove_html(x))
df['text'] = df['text'].apply(lambda x: remove_URL(x))
df['text'] = df['text'].apply(lambda x: remove_punct(x))
df['text'] = df['text'].apply(lambda x: correct_spellings(x))

df.to_csv('data/treated.csv')
