import json
import matplotlib.pyplot as plt
import numpy as np

from flask import Flask, jsonify, request
from model.ai import Ai

app = Flask(__name__)

@app.route("/train-ml", methods=['GET','POST'])
def train_ml():
  # TODO: ML TRAINING

  with open("data/fake.json", 'r') as f:
    txt = f.read()
    data = json.loads(txt)
    ai = Ai(data)
    # plot(ai.data)
    # print(ai.data)
    # reqData = request.get_json()
    return {}
  
def plot(data):
  X = data.index
  
  columnsArr = data.columns
  y_name = columnsArr[0]
  y = data[y_name].values

  sizes = np.random.uniform(15, 80, len(y))
  colors = np.random.uniform(15, 80, len(y))

  plt.scatter(x=X, y=y, s=sizes, c=colors)
  plt.xlabel("Últimos 100 dias")
  plt.ylabel("Preço em Dólares")
  plt.title(y_name)
  
  plt.show()

app.run(host='localhost', port=5000, debug=True)