import pandas as pd
import gensim.parsing.preprocessing as gsp
import re


filters = [
    gsp.strip_tags,
    gsp.strip_punctuation,
    gsp.strip_multiple_whitespaces,
    gsp.strip_numeric,
    gsp.remove_stopwords,
    gsp.strip_short,
    gsp.stem_text
]


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


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


def clean_text(text):
    text = text.lower()
    text = remove_URL(text)
    text = remove_emoji(text)
    for f in filters:
        text = f(text)
    return text


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

df = pd.concat([train, test])
df['text'] = df['text'].apply(lambda text: clean_text(text))
df[df['text'] == ''] = 'void'

df.to_csv('data/treated_2.csv')
