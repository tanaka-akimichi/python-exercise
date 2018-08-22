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
    x = x - np.max(x, axis=0)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    # if y.ndim == 1:
    #     t = t.reshape(1, t.size)
    #     y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=0)
             
    batch_size = y.shape[1]
    return -np.sum(np.log(y[t, np.arange(batch_size)] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

if __name__ == '__main__':

    x = np.array([[0, 1], [1, 2], [2, 4]])
    print(softmax(x))

    y = np.array([[1], [2], [2], [0]])
    t = np.array([[2], [1], [2], [-2]])
    print('mse={}'.format(mean_squared_error(y, t)))

    y = np.array([[0.2, 0.6], [0.5, 0.2], [0.3, 0.2]])
    t = np.array([[0, 1], [1, 0], [0, 0]])
    print(cross_entropy_error(y, t))
