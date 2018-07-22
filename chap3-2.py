import numpy as np
import sys
import matplotlib.pylab as plt

sys.path.append("C:\Users\tanaka.akimichi\PycharmProjects\dnn-from-scratch-exercise")

import chap3-5-2

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
y3 = np.tanh(x)/2.0+0.5 # Set the value range from 0 to 1.
y4 = relu(x)

plt.plot(x, y1, label='step')
plt.plot(x, y2, label='sigmoid')
plt.plot(x, y3, label='tanh')
plt.plot(x, y4, label='relu')

plt.title('Activation Function')

plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()
