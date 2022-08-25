# -*- coding: utf-8 -*-
import tensorflow as tf
import time

import random
import numpy as np
dataset = []
max_len = 10
batch_size = 4
num_train = 10000
query = np.random.randint(low=0, high=100, size=(batch_size, max_len))
title = np.random.randint(low=0, high=100, size=(batch_size, max_len))
label = np.random.randint(low=0, high=2, size=(batch_size, 1))
for _ in range(num_train):
  dataset.append((query, title, label))

encoder = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=100,
        output_dim=64,),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64,)
])

title_encoder = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=100,
        output_dim=64,),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64),
])

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

encoder_optimizer = tf.keras.optimizers.Adam(1e-4)
t_encoder_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(data):
  q, t, label = data
  with tf.GradientTape() as encoder_tape, tf.GradientTape() as title_encoder_tape:
    q_vec = encoder(q, training=True)
    t_vec = title_encoder(t, training=True)
    score = tf.reduce_sum(q_vec * t_vec, axis=-1, keepdims=True)
    loss = binary_cross_entropy(label, score)
    tf.print(loss)

  gradients_of_encoder = encoder_tape.gradient(loss, encoder.trainable_variables)
  gradients_of_title_encoder = title_encoder_tape.gradient(loss, title_encoder.trainable_variables)

  encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
  t_encoder_optimizer.apply_gradients(zip(gradients_of_title_encoder, title_encoder.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)


    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

train(dataset, 3)
