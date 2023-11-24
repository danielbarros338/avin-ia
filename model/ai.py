import pandas as pd
import tensorflow as tf

class Ai():
  def __init__(self, data):
    self.data = self.__formatData__(data)

  def __formatData__(self, data):
    dfData = []

    for df in data['timeSeriesDailyArr']:
      dfDataTemp = pd.DataFrame(data=df['Time Series (Daily)'])
      dfDataTemp = dfDataTemp.T

      dfDataTemp[dfDataTemp.columns] = dfDataTemp[dfDataTemp.columns].apply(pd.to_numeric, errors='coerce')
      dfDataTemp[dfDataTemp.columns] = tf.keras.utils.normalize(dfDataTemp[dfDataTemp.columns].values, axis=0)

      dfData.append(dfDataTemp)

    return dfData