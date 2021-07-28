import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from transformers import BertTokenizer, TFBertModel, TFTrainer, TFTrainingArguments

tf.keras.backend.set_floatx('float16')

# データを読み込む
data = pd.read_csv('data/treated.csv')

# 訓練用データを取得
train = data.dropna(subset=['target'])
train_text, train_label = train['text'], train['target']

# テスト用データを取得
test = data[data.isnull().any(1)]
test_text = test['text']

# BERT用のエンコーダーを読み込む
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# テキストを BERT 用にエンコーディング
train_encodings = tokenizer(train_text.to_list(),
                            padding='max_length',
                            truncation=False,
                            return_attention_mask=True,
                            max_length=100)

test_encoding = tokenizer(test_text.to_list(),
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
bert_model = TFBertModel.from_pretrained("bert-large-uncased")

# モデル構築
input = tf.keras.Input(shape=(100,), dtype='int32')
attention_masks = tf.keras.Input(shape=(100,), dtype='int32')
output = bert_model([input, attention_masks])
output = output[1]
output = tf.keras.layers.Dense(32, activation='relu')(output)
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
    [train_ids, train_att], train_label,
    epochs=2,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(monitor='loss', min_delta=0,
                      patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy',
                          patience=3,
                          verbose=1,
                          factor=0.5,
                          min_lr=0.0000001),
        ModelCheckpoint('models/bert.h5', save_best_only=True)
    ],
)

# 推論
predict = model.predict([test_ids, test_att])
opredict = np.round(predict).astype(int).reshape(-1)
test_df = pd.read_csv('data/test.csv')
sub = pd.DataFrame({'id': test_df['id'].values.tolist(), 'target': predict})
sub.to_csv('result/result_bert.csv', index=False)
