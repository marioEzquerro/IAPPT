import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras import datasets, layers, models
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing import image
from keras.utils import np_utils

NUMERIC_COLUMNS = ['gesto', 'indicep_Y','indicep_X','indicet_Y','indicet_X','corazonp_Y','corazonp_X', 'corazont_Y','corazont_X', 'anularp_Y','anularp_X', 'anulart_Y','anulart_X', 'meniquep_Y','meniquep_X','meniquet_Y','meniquet_X']
dtrain = pd.read_csv('training_data.csv', names=NUMERIC_COLUMNS)
deval = pd.read_csv('eval_data.csv')
y_train = dtrain.pop('gesto')
y_eval = deval.pop('gesto')



# # # Especificamos los campos
feature_columns = []
for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Creacion de la funcion de input
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_func = make_input_fn(dtrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_func = make_input_fn(deval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_func)  # train
result = linear_est.evaluate(eval_input_func)  # get model metrics/stats by testing on tetsing data

print(result['accuracy'])  