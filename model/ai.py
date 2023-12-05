import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow import keras
# from tensorflow.keras import layers

class Ai():
  def __init__(self, data):
    self.data = self.__formatData__(data)
    self.model, need_training = self.build_model()
    
    if need_training:
      self.training()

  def __formatData__(self, data):
    dfData = []

    for df in data['timeSeriesDailyArr']:
      self.__normalize_data__(df, dfData)

    self.labels = dfData[0].index

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

    # # Retirando o ano dos indices
    # dictIndex = {}
    # for i in range(0, len(dfDataTemp.index)):
    #   dictIndex[dfDataTemp.index[i]] = dfDataTemp.index[i][5:]

    # dfDataTemp.rename(index=dictIndex, inplace=True)
    dfDataTemp.index.name = 'Dates'

    arr.append(dfDataTemp)
  
  def training(self):
    highValues = self.data[0]['2. high'].values
    lowValues = self.data[0]['3. low'].values

    split_index = int(0.8 * len(highValues))
    train_data, test_data = highValues[:split_index], highValues[split_index:]

    X_train, y_train = np.array(train_data), np.array(lowValues[:split_index])
    X_test, y_test = np.array(test_data), np.array(lowValues[split_index:])

    print(X_train, y_train)

    self.history = self.model.fit(
      x=X_train,
      y=y_train,
      epochs=20,
      verbose=True
    )

    actual_dir = os.path.dirname(os.path.realpath(__file__))
    self.model.save(actual_dir + '\\ia-models\\future-stock.keras')

    predictions_normalized = self.model.predict(X_test)

    # Calcular MAE (Erro Absoluto Médio)
    mae_tf = tf.reduce_mean(tf.abs(y_test - predictions_normalized)).numpy()

    # Calcular MSE (Erro Quadrático Médio)
    mse_tf = tf.reduce_mean(tf.square(y_test - predictions_normalized)).numpy()

    # Calcular RMSE (Raiz Quadrada do Erro Quadrático Médio)
    rmse_tf = tf.sqrt(mse_tf)

    print(f"Prediction: {predictions_normalized}")
    print(f"MAE (TensorFlow): {mae_tf}")
    print(f"MSE (TensorFlow): {mse_tf}")
    print(f"RMSE (TensorFlow): {rmse_tf}")

  def build_model(self):
    actual_dir = os.path.dirname(os.path.realpath(__file__))
    model_exist = os.path.isfile(actual_dir +"\\ia-models\\future-stock.keras")
    
    if model_exist:
      return model_exist, False
    else:
      model = keras.Sequential([
        keras.layers.Dense(65, activation='relu', input_shape=[1]),
        keras.layers.Dense(37, activation='relu'),
        keras.layers.Dense(1, activation='linear')
      ])

      optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

      model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])

      return model, True
  
  def predict(self, x):
    return self.model.predict(x)

  def plot(self):
    # hist = pd.DataFrame(self.history.history)
    # hist['epoch'] = self.history.epoch
    # print(hist.tail())
    hist = pd.DataFrame(self.history.history)
    hist['epoch'] = self.history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Loss')
    plt.plot(hist['epoch'], hist['mse'], label = 'MSE')
    plt.plot(hist['epoch'], hist['mae'], label = 'MAE')
    plt.legend()
    plt.show()