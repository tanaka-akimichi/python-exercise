# coding: utf-8
# Change the vector to be a column vector, not a row vector.

import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import time

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# iters_num = 10000 # dist
iters_num = 6000
train_size = x_train.shape[0]  # number of samples
# batch_size = 100 # dist
batch_size = 200
learning_rate = 0.1  # dist
# learning_rate = 0.05

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

t1 = time.time()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    my_x_batch = x_batch.T  # 784(dimension) x 100(batch_size)
    t_batch = t_train[batch_mask]
    my_t_batch = t_batch.T  # 10(categories) x 100(batch_size)
    
    # 勾配
    # grad = network.numerical_gradient(my_x_batch, my_t_batch)
    grad = network.gradient(my_x_batch, my_t_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(my_x_batch, my_t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(my_x_batch, my_t_batch)
        test_acc = network.accuracy(x_test.T, t_test.T)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # print(train_acc, test_acc)
        # print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

t2 = time.time()
print('batch_size={}, iters_num={}, elapsed time={}'.\
      format(batch_size, iters_num, t2-t1))

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
# To DO
# plt.savefig(r'c:\Users\akimi\PycharmProjects\python-exercise\histgram.png') # the order is important.
plt.show()

x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='train loss')
plt.xlabel("iteration")
plt.ylim(0, 2.5)
plt.legend(loc='lower right')
# To DO
# plt.savefig(r'c:\Users\akimi\PycharmProjects\python-exercise\histgram.png') # the order is important.
plt.show()