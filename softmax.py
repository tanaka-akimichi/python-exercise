import numpy as np

a = np.array([1010, 1000, 990])
b = np.array([0.3, 2.9, 4.0])

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# print(softmax(a))
# print(softmax(b))

