import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/treated.csv')

train = data.dropna(subset=['target'])
train_text, train_label = train['text'], train['target']

test = data[data.isnull().any(1)]
test_text = test['text']

x_train, x_val, y_train, y_val =\
    train_test_split(train_text, train_label, test_size=0.15)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
train_ds = train_ds.batch(10)
val_ds = val_ds.batch(10)

for feat, targ in train_ds.take(5):
    print('Features: {}, Target: {}'.format(feat, targ))

preprocessing_layer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
    trainable=True)

input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
encoder_inputs = preprocessing_layer(input)
outputs = encoder(encoder_inputs)
net = outputs['pooled_output']
net = tf.keras.layers.Dropout(0.1)(net)
net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
model = Model(input, net)

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-6),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    # callbacks=[
    #     EarlyStopping(monitor='loss', min_delta=0,
    #                   patience=20, verbose=1),
    #     ReduceLROnPlateau(monitor='val_acc',
    #                       patience=3,
    #                       verbose=1,
    #                       factor=0.5,
    #                       min_lr=0.00001),
    #     ModelCheckpoint('models/best_v10.h5', save_best_only=True)
    # ],
)

# 推論
predict = model.predict(test_text)
opredict = np.round(predict).astype(int).reshape(3263)
test_df = pd.read_csv('data/test.csv')
sub = pd.DataFrame({'id': test_df['id'].values.tolist(), 'target': predict})
sub.to_csv('result/result_bert.csv', index=False)
