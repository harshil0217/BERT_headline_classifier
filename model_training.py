import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import keras_nlp
import numpy as np
import pickle
from official.nlp import optimization

#load encoded and preprocess model
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

#build classifier

def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_base_en_uncased", trainable=True)
    encoder_inputs = preprocessor(text_input)
    encoder = keras_nlp.models.BertBackbone.from_preset("bert_base_en_uncased")
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)


loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = [tf.metrics.BinaryAccuracy()]

#load in the datasets
with open('data/X_train.pkl', 'rb') as data:
    X_train = pickle.load(data)

with open('data/X_test.pkl', 'rb') as data:
    X_test = pickle.load(data)
    
with open('data/y_train.pkl', 'rb') as data:
    y_train = pickle.load(data)
    
with open('data/y_test.pkl', 'rb') as data:
    y_test = pickle.load(data)

#train the model

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = [tf.metrics.BinaryAccuracy()]


init_lr = 3e-5
optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)

classifier_model = build_classifier_model()

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

history = classifier_model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=5)