import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from transformers import BertTokenizer, TFBertModel, TFTrainer, TFTrainingArguments
from pandas import DataFrame

# データを読み込む
data = pd.read_csv('/content/treated.csv')

# 訓練用データを取得
train = data.dropna(subset=['target'])
train_texts, train_labels = train['text'], train['target']

# テスト用データを取得
test = data[data['target'].isnull()]
test_texts = test['text']

# BERT用のエンコーダーを読み込む
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# テキストを BERT 用にエンコーディング
train_encodings = tokenizer(train_texts.to_list(),
                            padding='max_length',
                            truncation=False,
                            return_attention_mask=True,
                            max_length=100)

test_encoding = tokenizer(test_texts.to_list(),
                          padding='max_length',
                          truncation=False,
                          return_attention_mask=True,
                          max_length=100)


def encode_tf_layers(encoding):
    dict_encoding = dict(encoding)
    input_id = np.asarray(dict_encoding['input_ids'])
    attention_id = np.asarray(dict_encoding['attention_mask'])
    return input_id, attention_id


# エンコードしたデータから単語IDとアテンションを取り出す
train_ids, train_att = encode_tf_layers(train_encodings)
test_ids, test_att = encode_tf_layers(test_encoding)

# BERT のモデルを読み込む
bert_model = TFBertModel.from_pretrained('bert-large-uncased')

# モデル構築
input = tf.keras.Input(shape=(100,), dtype='int32')
attention_masks = tf.keras.Input(shape=(100,), dtype='int32')
output = bert_model([input, attention_masks])
output = output[1]
output = tf.keras.layers.Dense(
    32, activation='relu', kernel_initializer='he_normal')(output)
output = tf.keras.layers.Dropout(0.2)(output)
output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
model = tf.keras.models.Model(
    inputs=[input, attention_masks], outputs=output)

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=6e-6),
    metrics=['accuracy']
)

history = model.fit(
    [train_ids, train_att], train_labels,
    epochs=2,
    batch_size=4,
    validation_split=0.2,
)

# 推論
predict = model.predict([test_ids, test_att])
predict = (predict > 0.5).astype(int).reshape(-1)

# CSV に出力
sample_data = pd.read_csv('/content/sample_submission.csv')
submit_data = DataFrame(
    {'id': sample_data['id'].to_list(), 'target': predict.tolist()})
submit_data.to_csv('/content/result.csv', index=False)
