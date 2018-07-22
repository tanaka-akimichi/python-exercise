# coding: utf-8
import matplotlib.pyplot as plt
# import sys, os
# sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from PIL import Image
from pylab import rcParams

from dataset.mnist import load_mnist


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# tmp = x_train[0] + x_train[1]
# print('shape={}'.format(x_train[0].shape))
# print('shape={}'.format(x_train[1].shape))
# print('shape={}'.format(tmp.shape))


# Extend the display size.
# rcParams['figure.figsize'] = 10, 10
rcParams['figure.figsize'] = 20, 20

def show_label_image(start, row_number, column_number):
    """
    Display the image from start in row_number rows and column_number columns.
    :param start: start position
    :param row_number: number of rows
    :param column_number: number of columns
    :return: None
    """

    fig = plt.figure()

    img = [0] * (row_number * column_number)
    label = [0] * (row_number * column_number)

    for i in range(row_number * column_number):
        # print(i)
        # img[i] = x_train[start + i]
        # label[i] = t_train[start + i]
        img[i] = x_test[start + i]
        label[i] = t_test[start + i]
        # print(label[i])
        # print(img[i].shape)
        img[i] = img[i].reshape(28, 28)   # 形状を元の画像サイズに変形
        # print(img[i].shape)

        ax1 = fig.add_subplot(row_number, column_number, i+1)

        # No labels and no ticks.
        ax1.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                        length=0)
        ax1.imshow(img[i])
        ax1.set_title(label[i])

    plt.show()

i = input("Please Enter Number(i): ")
r = 5
c = 10
# r = input("Please Enter Number(r): ")
# c = input("Please Enter Number(c): ")
show_label_image(int(i), r, c)

# plt.show()



