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
    preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_base_en_uncased",trainable=True)
    encoder_inputs = preprocessor(text_input)
    encoder = keras_nlp.models.BertBackbone.from_preset("bert_base_en_uncased")
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)


loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

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
epochs = 5
steps_per_epoch = len(X_train)//32
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')





