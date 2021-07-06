import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from transformers import BertJapaneseTokenizer, AutoTokenizer
from transformers import TFBertModel


def tokenize_map_fn(tokenizer, max_length=128):
    """map function for pretrained tokenizer"""
    def _tokenize(text_a, label):
        inputs = tokenizer.encode_plus(
            text_a.numpy().decode('utf-8'),
            add_special_tokens=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        return input_ids, token_type_ids, attention_mask, label

    def _map_fn(text, label):
        out = tf.py_function(_tokenize, inp=[text, label], Tout=(
            tf.int32, tf.int32, tf.int32, tf.int32))
        return (
            {"input_ids": out[0], "token_type_ids": out[1],
                "attention_mask": out[2]},
            out[3]
        )

    return _map_fn


def load_dataset(data, tokenizer, max_length=128, train_batch=32):
    train_dataset = data.map(tokenize_map_fn(tokenizer, max_length=max_length))
    train_dataset = train_dataset.shuffle(train_batch).padded_batch(train_batch, padded_shapes=(
        {'input_ids': [-1], 'token_type_ids': [-1], 'attention_mask': [-1]}, []))
    return train_dataset


BATCH_SIZE = 128


@tf.function
def mapping(text, label):
    return text, label


text_datasets = []
text_dir = "C:/Users/owner/Desktop/NLP/DATA/livedoor-lite"
for d in tf.compat.v1.gfile.ListDirectory(text_dir):
    data_dir = os.path.join(text_dir, d)
    label = int(d == "dokujo-tsushin")  # ディレクトリがdokujo-tsushinだったらTrue
    if os.path.isdir(data_dir):
        for file_name in tf.compat.v1.gfile.ListDirectory(data_dir):
            text_dataset = tf.data.TextLineDataset(os.path.join(
                data_dir, file_name)).map(lambda ex: mapping(ex, label))
            text_datasets.append(text_dataset)

# 分かち書きせずテキスト単位の配列に変換する
new_dataset = text_datasets[0]
for text_dataset in text_datasets[1:]:
    new_dataset = new_dataset.concatenate(text_dataset)


max_length = 128


tokenizer = BertJapaneseTokenizer.from_pretrained(
    'cl-tohoku/bert-base-japanese-v2')
train_dataset = load_dataset(
    new_dataset, tokenizer, max_length=max_length, train_batch=BATCH_SIZE)


input_ids = tf.keras.layers.Input(
    shape=(max_length, ), dtype='int32', name='input_ids')
attention_mask = tf.keras.layers.Input(
    shape=(max_length, ), dtype='int32', name='attention_mask')
token_type_ids = tf.keras.layers.Input(
    shape=(max_length, ), dtype='int32', name='token_type_ids')
inputs = [input_ids, attention_mask, token_type_ids]

bert = TFBertModel.from_pretrained(
    'cl-tohoku/bert-base-japanese-char-whole-word-masking')
bert.trainable = False
x = bert(inputs)

out = x[1]

fully_connected = tf.keras.layers.Dense(128, activation='relu')(out)
Y = tf.keras.layers.Dense(1, activation='sigmoid')(fully_connected)

model = tf.keras.Model(inputs=inputs, outputs=Y)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-5))

model.summary()
model.fit(train_dataset, epochs=50)

sample_pred_text = '大谷翔平、ボール三連続「思ってもみませんでした」'
encoded = tokenizer.encode_plus(
    sample_pred_text,
    sample_pred_text,
    add_special_tokens=True,
    max_length=128,
    pad_to_max_length=True,
    return_attention_mask=True
)

inputs = {"input_ids": tf.expand_dims(encoded["input_ids"], 0),
          "token_type_ids": tf.expand_dims(encoded["token_type_ids"], 0),
          "attention_mask": tf.expand_dims(encoded["attention_mask"], 0)
          }

res = model.predict_on_batch(inputs)
res.numpy()
