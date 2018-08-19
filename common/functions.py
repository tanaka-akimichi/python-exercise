# Modify for column vectors input

# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if len(x.shape) >= 2:
        num = x.shape[1]
        y = np.empty_like(x).T
        for i in range(num):
            x[:, i] = x[:, i] - np.max(x[:, i])  # オーバーフロー対策
            y_tmp = np.exp(x[:, i]) / np.sum(np.exp(x[:, i]))
            # y[:, i] = y_tmp[:,np.newaxis]
            y[i] = y_tmp
        return y
    else:
        x = x - np.max(x, axis=0)   # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x), axis=0)


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

if __name__ == '__main__':

    # x = np.array([0.1, 0.2, 0.4, 0.1, 0.1])
    # x = np.array([[0.1, 0.2], [0.3, 0.5], [0.2, 0.6]])
    x = np.array([[0, 1], [1, 2], [2, 3]])
    print(softmax(x))
