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
from sklearn.model_selection import train_test_split
import gensim.downloader as api
from tqdm import tqdm
from tensorflow.keras.initializers import Constant


# nltk.download('punkt')


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()


df = pd.read_csv('data/treated.csv')


# 文章を単語リストに変換する
def texts_to_words(texts):
    stop = stopwords.words('english')
    corpus = []
    for text in tqdm(texts):
        words = [word for word in word_tokenize(
            text) if word.isalpha() & (word not in stop)]
        corpus.append(words)
    return corpus


# 文章を単語リスト変換
corpus = texts_to_words(df['text'])

# word2vecの辞書データをロード
wv = api.load('glove-twitter-100')

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
    try:
        emb_vec = wv.get_vector(word)
    except KeyError:
        pass
    else:
        embedding_matrix[i] = emb_vec


# padding処理。50単語に満たないものは0埋めする。
MAX_LEN = 50
texts_pad = pad_sequences(sequences, maxlen=MAX_LEN,
                          truncating='post', padding='post')


# 訓練用データと評価用データとテストデータに分ける
train_df = pd.read_csv('data/train.csv')
train_text = texts_pad[:train_df['text'].shape[0]]
train_label = train_df['target']
test_text = texts_pad[train_df['text'].shape[0]:]
X_train, X_test, y_train, y_test =\
    train_test_split(train_text, train_label, test_size=0.15)


# 埋め込み層を含むモデルを作成
model = Sequential([
    Embedding(num_words, 100,
              embeddings_initializer=Constant(embedding_matrix),
              input_length=MAX_LEN,
              trainable=False),
    SpatialDropout1D(0.25),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(128)),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train, batch_size=4, epochs=7,
    validation_data=(X_test, y_test),
    callbacks=[
        EarlyStopping(monitor='loss', min_delta=0,
                      patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy',
                          patience=3,
                          verbose=1,
                          factor=0.5,
                          min_lr=1e-6),
        ModelCheckpoint('models/w2v_model.h5',
                        save_best_only=True)
    ],
)

# 推論
predict = model.predict(test_text)
predict = np.round(predict).astype(int).reshape(3263)
test_df = pd.read_csv('data/test.csv')
sub = pd.DataFrame({'id': test_df['id'].values.tolist(), 'target': predict})
sub.to_csv('result/result.csv', index=False)
