from pandas.core.frame import DataFrame
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, SpatialDropout1D, Conv1D, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tqdm import tqdm
from tensorflow.keras.initializers import Constant

# データ読み込み
data = pd.read_csv('data/treated.csv')

# 訓練用データを取得
train = data.dropna(subset=['target'])
train_texts, train_labels = train['text'], train['target']

# テスト用データを取得
test = data[data.isnull().any(1)]
test_texts = test['text']

# 埋め込み用の辞書を取得
embedding_dict = {}
with open('C:/Users/owner/Desktop/NLP/DATA/glove/glove.6B.100d.txt', 'r', encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding_dict[word] = vectors


# トークン化する処理
def text_to_tokens(texts):
    stop = stopwords.words('english')
    tokens = []
    for text in tqdm(texts):
        words = [word.lower() for word in word_tokenize(text)
                 if(word.isalpha() & (word not in stop))]
        tokens.append(words)
    return tokens


# テキストを単語の配列にする
train_tokens = text_to_tokens(train_texts)
test_tokens = text_to_tokens(test_texts)

# 単語を扱いやすいように番号に変換する辞書を作る
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(test_tokens + train_tokens)

# 単語を番号に変換する
train_sequences = tokenizer_obj.texts_to_sequences(train_tokens)
test_sequences = tokenizer_obj.texts_to_sequences(test_tokens)

# 単語:番号 の辞書
word_to_index = tokenizer_obj.word_index

# ユニークな単語数を取得しておく
num_words = len(word_to_index) + 1

# 埋め込み用のベクトルを用意
embedding_matrix = np.zeros((num_words, 100))

# 単語を回しながらインデックスに対応したベクトル表現の配列を作成する
for word, i in tqdm(word_to_index.items()):
    emb_vec = embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i] = emb_vec

# 1文章の最大単語数
MAX_LEN = 50

# padding処理。50単語に満たないものは0埋めする。
train_pad = pad_sequences(train_sequences, maxlen=MAX_LEN,
                          truncating='post', padding='post')
test_pad = pad_sequences(test_sequences, maxlen=MAX_LEN,
                         truncating='post', padding='post')

# モデル構築
model = Sequential([
    Embedding(num_words, 100, embeddings_initializer=Constant(
        embedding_matrix), input_length=MAX_LEN, trainable=False),
    SpatialDropout1D(0.2),
    LSTM(64),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dense(1, activation='sigmoid')
])
model.summary()
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 学習
model.fit(
    train_pad, train_labels,
    epochs=7,
    validation_split=0.2,
)

# 推論
predict = model.predict(test_pad)
predict = (predict > 0.5).astype('int32').reshape(3263)
print(predict)
print(predict.shape)

# CSV に出力
sample_data = pd.read_csv('data/sample_submission.csv')
submit_data = DataFrame(
    {'id': sample_data['id'].to_list(), 'target': predict.tolist()})
submit_data.to_csv('result/result.csv')
