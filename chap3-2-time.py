import numpy as np
import matplotlib.pylab as plt

start = -10
end = 10

from stop_watch import stop_watch

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

@stop_watch
def n_step_functions(n):
    for x in np.arange(start, end, (end - start) / n):
        step_function(x)

@stop_watch
def n_sigmoid(n):
    for x in np.arange(start, end, (end-start)/n):
        sigmoid(x)

@stop_watch
def n_relu(n):
    for x in np.arange(start, end, (end-start)/n):
        step_function(x)

# n_step_functions(10000000)
# n_sigmoid(10000000)
# n_relu(10000000)

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
