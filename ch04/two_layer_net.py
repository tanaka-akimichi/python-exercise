# coding: utf-8
# Change the vector to be a column vector, not a row vector.

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(hidden_size, input_size)
        row_b1 = np.zeros(hidden_size)
        self.params['b1'] = row_b1[:, np.newaxis]
        self.params['W2'] = weight_init_std * np.random.randn(output_size, hidden_size)
        row_b2 = np.zeros(output_size)
        self.params['b2'] = row_b2[:, np.newaxis]

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(W1, x) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(W2, z1) + b2
        y = softmax(a2)
        
        return y
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(W1, x) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(W2, z1) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1, dy.T)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(W2, dy.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(da1, x.T)
        grads['b1'] = np.sum(da1, axis=0)

        return grads


if __name__ == '__main__':

    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)
    print(net.params['b1'].shape)
    print(net.params['W2'].shape)
    print(net.params['b2'].shape)

    # x = np.random.rand(784)
    # my_x = x[:, np.newaxis]
    # y = net.predict(my_x)

    x = np.random.rand(784, 100)
    y = net.predict(x)
