import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

class Ai():
  def __init__(self, data):
    self.data = self.__formatData__(data)
    self.model = self.build_model()

  def __formatData__(self, data):
    dfData = []

    for df in data['timeSeriesDailyArr']:
      dfDataTemp = pd.DataFrame(data=df['Time Series (Daily)'])
      dfDataTemp = dfDataTemp.T

      dfDataTemp[dfDataTemp.columns] = dfDataTemp[dfDataTemp.columns].apply(pd.to_numeric, errors='coerce')
      dfDataTemp[dfDataTemp.columns] = tf.keras.utils.normalize(dfDataTemp[dfDataTemp.columns].values, axis=0)

      dfData.append(dfDataTemp)

    return dfData
  
  def trainig(self):
    train_dataset = self.data[0].sample(frac=0.8, random_state=0)
    test_dataset = self.data[0].drop(train_dataset.index)

    train_labels = train_dataset.pop('1. open')
    test_labels = test_dataset.pop('1. open')

    # TODO: FINISH the training

    return train_labels, test_labels

  def build_model(self):
    model = keras.Sequential([
      layers.dense(100, activation='relu', input_shape=[80]),
      layers.dense(200, activation='relu'),
      layers.dense(1)
    ])

    optimizer = tf.keras.optimizer.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model