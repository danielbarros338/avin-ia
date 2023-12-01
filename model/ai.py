import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
# from tensorflow.keras import layers

class Ai():
  def __init__(self, data):
    self.data = self.__formatData__(data)
    # self.model = self.build_model()

  def __formatData__(self, data):
    dfData = []

    for df in data['timeSeriesDailyArr']:
      self.__normalize_data__(df, dfData)

    return dfData
  
  def __normalize_data__(self, df, arr=[]):
    """
    Normalize the data

    Input:
      - df: Dict of data
      - arr: List will receive the Dataframes
    """
    dfDataTemp = pd.DataFrame(data=df['Time Series (Daily)'])
    dfDataTemp = dfDataTemp.T

    # Normalizando valores entre 0 e 1
    dfDataTemp[dfDataTemp.columns] = dfDataTemp[dfDataTemp.columns].apply(pd.to_numeric, errors='coerce')
    dfDataTemp[dfDataTemp.columns] = tf.keras.utils.normalize(dfDataTemp[dfDataTemp.columns].values, axis=0)

    # Retirando o ano dos indices
    dictIndex = {}
    for i in range(0, len(dfDataTemp.index)):
      dictIndex[dfDataTemp.index[i]] = dfDataTemp.index[i][5:]

    dfDataTemp.rename(index=dictIndex, inplace=True)
    dfDataTemp.index.name = 'Dates'

    arr.append(dfDataTemp)

  
  def training(self):
    openValues = self.data[0]['1. open']
    split_index = int(0.8 * len(openValues))
    train_data, test_data = openValues.iloc[:split_index], openValues[split_index:]

    print(openValues.columns)
    print(train_data.shape, test_data.shape)

  def build_model(self):
    model = keras.Sequential([
      keras.layers.Input(shape=(5,)),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(32, activation='relu'),
      keras.layers.Dense(5, activation='linear')
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    return model
  
  def predict(self):
    self.training()
    # example_batch = self.training_test['train_dataset'][:10]
    # example_result = self.model.predict(example_batch)
    # print('result', example_result)

  def plot(self):
    # hist = pd.DataFrame(self.history.history)
    # hist['epoch'] = self.history.epoch
    # print(hist.tail())
    hist = pd.DataFrame(self.history.history)
    hist['epoch'] = self.history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['loss'], hist['epoch'],
            label='Loss')
    plt.plot(hist['accuracy'], hist['epoch'],
            label = 'Accuracy')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['loss'], hist['epoch'],
            label='Loss')
    plt.plot(hist['accuracy'], hist['epoch'],
            label = 'Accuracy')
    plt.ylim([0,20])
    plt.legend()
    plt.show()