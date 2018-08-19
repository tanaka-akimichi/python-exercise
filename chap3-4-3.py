"""
Both codes of row vector times matrix and matrix times column vector
are provided.
As a method of converting a row vector to a column vector,
  method 1: using x [:, np.newaxis],
  method 2: a method using np.matrix () and T (transpose)
are used.
Comparing the processing speed of the original method with the method 1
and method 2, the results show that the original method and the method 1
are almost the same, and the method 2 is very slow.
"""

import numpy as np
import timeit

# Repetition number of time measurement
repeat = 1000000


def identity_function(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""
Original method
Copy the source code on the textbook on Page 65.
row vector x matrix
"""


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print("original value={}".format(y))  # [ 0.31682708  0.69627909]

# Measure processing time.
result1 = timeit.timeit('forward(network, x)',
                        globals=globals(), number=repeat)
print("elapsed time(row_vector x matrix) = {}".format(result1))

"""
Method 1
Matrix x column vector
Convert row vector to column vector using x[:, np.newaxis]
"""


def my_init_network():
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


def my_forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(W1, x) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(W2, z1) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(W3, z2) + b3
    y = identity_function(a3)

    return y

my_network = my_init_network()
my_x = x[:, np.newaxis]
y = my_forward(my_network, my_x)
print("my value={}".format(y))
# [[ 0.31682708]
# [ 0.69627909]]

# Measure processing time.
result2 = timeit.timeit('my_forward(my_network, my_x)',
                        globals=globals(), number=repeat)
print("elapsed time(matrix x column_vector) = {}".format(result2))


"""
Method 2
Matrix x column vector
Convert row vector to column vector using np.matrix and x.T
Much slower than original method and method 1.
"""


def matrix_init_network():
    network = {}
    network['W1'] = np.matrix([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    network['b1'] = np.matrix([0.1, 0.2, 0.3]).T
    network['W2'] = np.matrix([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    network['b2'] = np.matrix([0.1, 0.2]).T
    network['W3'] = np.matrix([[0.1, 0.2], [0.3, 0.4]])
    network['b3'] = np.matrix([0.1, 0.2]).T

    return network


def matrix_forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(W1, x) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(W2, z1) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(W3, z2) + b3
    y = identity_function(a3)

    return y

matrix_network = matrix_init_network()
matrix_x = np.matrix([1.0, 0.5]).T
y = matrix_forward(matrix_network, matrix_x)
print("matrix value={}".format(y))
# [[ 0.31682708]
# [ 0.69627909]]

# Measure processing time.
result3 = timeit.timeit('my_forward(matrix_network, matrix_x)',
                        globals=globals(), number=repeat)
print("elapsed time(matrix x column_vector using np.matrix) = {}"
      .format(result3))

"""
original value=[ 0.31682708  0.69627909]
elapsed time(row_vector x matrix) = 9.295675532992151
my value=[[ 0.31682708]
 [ 0.69627909]]
elapsed time(matrix x column_vector) = 9.202078493035662
matrix value=[[ 0.31682708]
 [ 0.69627909]]
elapsed time(matrix x column_vector using np.matrix) = 25.460554655446654
"""
