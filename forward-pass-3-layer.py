# Stanford CS231n Convolutional Neural Networks for Visual Recognition
import numpy as np
import timeit

# Sample code from their home page.
# forward-pass of a 3-layer neural network:
# f = lambda x: 1.0/(1.0 + np.exp(-x)) # activation function (use sigmoid)
# x = np.random.randn(3, 1) # random input vector of three numbers (3x1)
# h1 = f(np.dot(W1, x) + b1) # calculate first hidden layer activations (4x1)
# h2 = f(np.dot(W2, h1) + b2) # calculate second hidden layer activations (4x1)
# out = np.dot(W3, h2) + b3 # output neuron (1x1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


def Fei_Fei_init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    row_b1 = np.array([0.1, 0.2, 0.3])
    network['b1'] = row_b1[:, np.newaxis]
    network['W2'] = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    row_b2 = np.array([0.1, 0.2])
    network['b2'] = row_b2[:, np.newaxis]
    network['W3'] = np.array([[0.1, 0.2], [0.3, 0.4]])
    row_b3 = np.array([0.1, 0.2])
    network['b3'] = row_b3[:, np.newaxis]

    return network


def Fei_Fei_forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(W1, x) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(W2, z1) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(W3, z2) + b3
    y = identity_function(a3)

    return y

# Repetition number of time measurement
repeat = 1000000

Fei_Fei_network = Fei_Fei_init_network()
Fei_Fei_x = np.array((2, 1))
Fei_Fei_x = [[1.0], [0.5]]

y = Fei_Fei_forward(Fei_Fei_network, Fei_Fei_x)
print("my value={}".format(y))

# Measure processing time.
result = timeit.timeit('Fei_Fei_forward(Fei_Fei_network, Fei_Fei_x)',
                        globals=globals(), number=repeat)
print("elapsed time(matrix x column_vector) = {}".format(result))


# [[ 0.31682708]
# [ 0.69627909]]
# Repetition number of time measurement
# repeat = 1000000
# elapsed time(matrix x column_vector) = 9.376588585762939