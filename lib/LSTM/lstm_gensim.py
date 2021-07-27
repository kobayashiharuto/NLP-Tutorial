from gensim.models import keyedvectors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import gensim.downloader as api
from tqdm import tqdm
from gensim.models import KeyedVectors

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
    # ストップワードを取得
    stop = stopwords.words('english')
    corpus = []
    for text in tqdm(texts):
        words = [word.lower() for word in word_tokenize(text) if(
            (word.isalpha() == 1) & (word not in stop))]
        corpus.append(words)
    return corpus


# 文章を単語リスト変換
corpus = texts_to_words(df['text'])

# word2vecの辞書データをダウンロード
wv = api.load('word2vec-google-news-300')

# シーケンス番号に対しての辞書を作成する
vocabulary = wv.key_to_index
tokenizer = Tokenizer(num_words=len(vocabulary))
tokenizer.word_index = vocabulary
sequences = tokenizer.texts_to_sequences(corpus)

print(tokenizer.texts_to_sequences(['i', 'am', 'note']))

wv.save_word2vec_format('w2v/word2vec.kv')
wv = KeyedVectors.load_word2vec_format('wv2/word2vec.kv')
# keras の embedding層に変換
embedding_layer = wv.get_keras_embedding()

# padding処理。50単語に満たないものは0埋めする。
texts_pad = pad_sequences(sequences, maxlen=50,
                          truncating='post', padding='post')

# 埋め込み層を含むモデルを作成
model = Sequential([
    embedding_layer,
    SpatialDropout1D(0.2),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

optimzer = Adam(learning_rate=1e-5)

model.compile(loss='binary_crossentropy',
              optimizer=optimzer, metrics=['accuracy'])

model.summary()

train_df = pd.read_csv('data/train.csv')
train_text = texts_pad[:train_df['text'].shape[0]]
train_label = train_df['target']
test_text = texts_pad[train_df['text'].shape[0]:]

X_train, X_test, y_train, y_test = train_test_split(
    train_text, train_label, test_size=0.15)

history = model.fit(X_train, y_train, batch_size=4, epochs=15,
                    validation_data=(X_test, y_test))


predict = model.predict(test_text)
predict = np.round(predict).astype(int).reshape(3263)
test_df = pd.read_csv('data/test.csv')
sub = pd.DataFrame({'id': test_df['id'].values.tolist(), 'target': predict})
sub.to_csv('result/result.csv', index=False)
