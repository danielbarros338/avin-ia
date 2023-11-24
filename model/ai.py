import pandas as pd
import tensorflow as tf

class Ai():
  def __init__(self, data):
    self.data = self.__formatData__(data)

  def __formatData__(self, data):
    dfData = data['timeSeriesDaily']['Time Series (Daily)']
    dfData = pd.DataFrame(data=dfData)
    dfData = dfData.T

    dfData[dfData.columns] = dfData[dfData.columns].apply(pd.to_numeric, errors='coerce')
    dfData[dfData.columns] = tf.keras.utils.normalize(dfData[dfData.columns].values, axis=0)

    print(dfData.head())

    return dfData