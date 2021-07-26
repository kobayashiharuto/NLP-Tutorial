from pandas.core.frame import DataFrame
from pandas.core.reshape.concat import concat
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from tensorflow.python.keras.backend import constant
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()


df = pd.read_csv('data/treated.csv')


def create_corpus(texts):
    corpus = []
    for text in tqdm(texts):
        words = [word.lower() for word in word_tokenize(text) if(
            (word.isalpha() == 1) & (word not in stopwords))]
        corpus.append(words)
    return corpus


# 文章をトークン化する
corpus = create_corpus(df['text'])

# 埋め込み用の辞書を取得
embedding_dict = {}
with open('C:/Users/owner/Desktop/NLP/DATA/glove/glove.6B.100d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding_dict[word] = vectors

# テキストの単語を扱いやすいようにシーケンス番号に変換
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences = tokenizer_obj.texts_to_sequences(corpus)

# 単語 : シーケンス番号の辞書
word_index = tokenizer_obj.word_index

# ユニークな単語数を取得しておく
num_words = len(word_index) + 1

# 埋め込み用のベクトルを用意
embedding_matrix = np.zeros((num_words, 100))

# 単語を回しながらシーケンス番号に対応したベクトル表現のリストを作成する
for word, i in tqdm(word_index.items()):
    if i > num_words:
        print(f'Really? {word} is No.{i}')
        continue

    emb_vec = embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i] = emb_vec

# 1文章の最大単語数
MAX_LEN = 50

# padding処理。50単語に満たないものは0埋めする。
texts_pad = pad_sequences(sequences, maxlen=MAX_LEN,
                          truncating='post', padding='post')

# 埋め込み層を含むモデルを作成
model = Sequential([
    Embedding(num_words, 100, embeddings_initializer=constant(
        embedding_matrix), input_length=MAX_LEN, trainable=False),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])


optimzer = Adam(learning_rate=1e-5)

model.compile(loss='binary_crossentropy',
              optimizer=optimzer, metrics=['accuracy'])

model.summary()
