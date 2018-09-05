import math
import numpy as np
import matplotlib.pylab as plt

def function_1(x):
    return math.cos(math.sin(x))

def function_1_diff(x):
    return -math.sin(math.sin(x)) * math.cos(x)

def numerical_diff_2(f, x, h=1e-4):
    return (f(x + h) - f(x)) / h

def numerical_diff_3(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_diff_5(f, x, h=1e-4):
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)

def numerical_diff_7(f, x, h=1e-4):
    return (f(x + 3*h) - 9*f(x + 2*h) + 45*f(x + h) \
            - 45*f(x - h) + 9*f(x - 2*h) - f(x - 3*h)) / (60 * h)

if __name__ == '__main__':
    """
    Evaluate the error of numerical differentiation.
    The target function is
        y = cos (sinx), x = PI / 4
    Perform 2 point, 3 point, 5 point and 7 point approximation.
    The smaller the step h, the smaller the truncation error but the larger 
    the effect of the rounding error.
    Reference: https://na-inet.jp/nasoft/chap15.pdf
    """

    x0 = math.pi / 4

    # print(function_1_diff(x0))
    # print(numerical_diff_2(function_1, x0))
    # print(numerical_diff_3(function_1, x0))
    # print(numerical_diff_5(function_1, x0))
    # print(numerical_diff_7(function_1, x0))

    i = 0
    x1 = np.zeros(21)
    y2 = np.zeros(21)
    y3 = np.zeros(21)
    y5 = np.zeros(21)
    y7 = np.zeros(21)
    real_value = function_1_diff(x0)
    for x in range(0, 21):
        h = 2 ** (-x)
        # print(h)

        x1[i] = h
        y2[i] = np.fabs(numerical_diff_2(function_1, x0, h=h) - real_value)
        y3[i] = np.fabs(numerical_diff_3(function_1, x0, h=h) - real_value)
        y5[i] = np.fabs(numerical_diff_5(function_1, x0, h=h) - real_value)
        y7[i] = np.fabs(numerical_diff_7(function_1, x0, h=h) - real_value)

        i += 1

    # print('x1:\n{}'.format(x0))
    # print('-----')
    # print('y2:\n{}'.format(y2))
    # print('-----')
    # print('y3:\n{}'.format(y3))
    # print('-----')
    # print('y5:\n{}'.format(y5))
    # print('-----')
    # print('y7:\n{}'.format(y7))

    plt.xscale("log")
    plt.yscale("log")
    ax = plt.gca()
    ax.invert_xaxis()
    ax.set_xlabel('h')
    ax.set_ylabel('error')
    plt.plot(x1, y2, label="two points")
    plt.plot(x1, y3, label="three points")
    plt.plot(x1, y5, label="five points")
    plt.plot(x1, y7, label="seven points")
    plt.legend(loc='upper right')
    plt.show()


