import pandas as pd
import numpy as np
import tensorflow as tf


def generate_random_points(n, x_max=3000, y_max=1000):
  x_loc = np.random.random(n) * x_max
  y_loc = np.random.random(n) * y_max
  return np.array([x_loc, y_loc]).T

def add_pickup_and_dropoff(df_in):
  df = df_in.copy()
  df[['pickup_x', 'pickup_y']] = generate_random_points(len(df))
  df[['dropoff_x', 'dropoff_y']] = generate_random_points(len(df))
  return df

def shipping_fee_func(df_in, min_weight = 2., min_fee = 5., rate = 0.001):
  return np.maximum(np.maximum(df_in['weight'], min_weight) * df_in['distance'] * rate, min_fee)

def create_dataframe(x, y):
  df = pd.DataFrame({'mnist_label': y})
  df['image'] = [x[i,:,:] for i in range(x.shape[0])]
  df['weight'] = df['mnist_label'] + np.random.random(len(df)) - 0.5
  df = add_pickup_and_dropoff(df)
  df['distance'] = np.sqrt((df['dropoff_x'] - df['pickup_x'])**2 + (df['dropoff_y'] - df['pickup_y'])**2)
  df['shipping_fee'] = shipping_fee_func(df)
  return df

def create_synthetic_data():
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  df_train = create_dataframe(x_train, y_train)
  df_test = create_dataframe(x_test, y_test)

  df_train['train_test'] = 'train'
  df_test['train_test'] = 'test'

  df = pd.concat((df_train, df_test))
  return df

