import numpy as np
from random import shuffle
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

class neuron:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    
    self.num_features = 4
    self.m = x.shape[0]
    
    self.layer0_in_dim = 4
    self.layer0_out_dim = 20
    
    self.layer1_in_dim = 20
    self.layer1_out_dim = 5
    
    self.layer2_in_dim = 5
    self.layer2_out_dim = 1
    
    self.w0 = np.random.randn(self.layer0_out_dim, self.layer0_in_dim) * 0.1
    self.b0 = np.zeros([self.layer0_out_dim, 1])
    self.f1 = np.zeros(self.b0.shape)
    
    self.w1 = np.random.randn(self.layer1_out_dim, self.layer1_in_dim) * 0.1
    self.b1 = np.zeros([self.layer1_out_dim, 1])
    self.f2 = np.zeros(self.b1.shape)
    
    self.w2 = np.random.randn(self.layer2_out_dim, self.layer2_in_dim) * 0.1
    self.b2 = np.zeros(self.layer2_out_dim)
    self.f3 = np.zeros(self.b2.shape)
  
  def forward(self, x_per):
    self.f1 = np.dot(self.w0, x_per.T) + self.b0
    self.f2 = np.dot(self.w1, self.sigmoid(self.f1)) + self.b1
    self.f3 = np.dot(self.w2, self.sigmoid(self.f2)) + self.b2
    return self.sigmoid(self.f3)
  
  def backward(self, y_predict, y_per, x_per, lr):
    grad0 = y_predict - y_per
    
    grad1 = np.dot(grad0.reshape(-1, 1), self.w2) * self.dev_sigmoid(self.f2.T)
    
    grad2 = np.dot(grad1, self.w1) * self.dev_sigmoid(self.f1.T)
    
    self.w2 = self.w2 - lr * (np.dot(grad0.reshape(1, -1), self.sigmoid(self.f2.T))) / self.m
    self.b2 = self.b2 - lr * np.sum(grad0) / self.m
    
    self.w1 = self.w1 - lr * np.dot(grad1.T, self.sigmoid(self.f1.T)) / self.m
    self.b1 = self.b1 - lr * np.sum(grad1.T, axis=1, keepdims=True) / self.m
    
    self.w0 = self.w0 - lr * np.dot(grad2.T, x_per)
    self.b0 = self.b0 - lr * np.sum(grad2.T, axis=1, keepdims=True)
  
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def dev_sigmoid(self, x):
    return self.sigmoid(x) * (1 - self.sigmoid(x))
  
  def train(self, epoches, learning_rate=0.01):
    for epoch in range(epoches):
      y_predict = self.forward(self.x)
      self.backward(y_predict, self.y, self.x, learning_rate)

def data_shuffle(dataX, dataY):
  data_inx = [i for i in range(len(dataY))]
  shuffle(data_inx)

  return dataX[data_inx], dataY[data_inx]
  
if __name__ == "__main__":
  iris_data = load_iris()
  X, Y = iris_data['data'][:100], iris_data['target'][:100]
  NewX, NewY = data_shuffle(X, Y)
  
  trainX, trainY = NewX[:80], NewY[0:80]
  testX, testY = NewX[80:], NewY[80:]

  network = neuron(trainX, trainY)
  network.train(epoches=200, learning_rate=0.1)
  py = network.forward(testX)[0]
  print(py)
  py[py >= 0.5] = 1
  py[py < 0.5] = 0
  print(accuracy_score(py, testY))