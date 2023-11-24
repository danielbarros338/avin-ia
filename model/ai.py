import pandas as pd

class Ai():
  def __init__(self, data):
    self.data = self.__formatData__(data)

  def __formatData__(self, data):
    dfData = data['timeSeriesDaily']['Time Series (Daily)']
    dfData = pd.DataFrame(data=dfData)
    dfData = dfData.T

    return dfData