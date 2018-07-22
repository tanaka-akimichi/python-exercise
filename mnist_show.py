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
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# Extend the display size.
# rcParams['figure.figsize'] = 10, 10
rcParams['figure.figsize'] = 20, 20  # for ASUS Note PC

def show_label_image(start, row_number, column_number, data_type):
    """
    Display the image from start in row_number rows and column_number columns.
    :param start: start position
    :param row_number: number of rows
    :param column_number: number of columns
    :param data_type: train or test
    :return: None
    """
    if data_type != 'train' and data_type != 'test':
        print("usage: show_label_image(start, row_number, column_number, "
              "data_type)")
        print("data_type must be train or test. ({}).".format(data_type))
        return

    fig = plt.figure()

    img = [0] * (row_number * column_number)
    label = [0] * (row_number * column_number)

    for i in range(row_number * column_number):
        ax1 = fig.add_subplot(row_number, column_number, i+1)
        if data_type == 'train':
            img[i] = x_train[start + i]
            label[i] = t_train[start + i]
            ax1.set_title('train {} ({})'.format(start+i, str(label[i])))
        else:
            img[i] = x_test[start + i]
            label[i] = t_test[start + i]
            ax1.set_title('test {} ({})'.format(start+i, str(label[i])))
        img[i] = img[i].reshape(28, 28)   # 形状を元の画像サイズに変形

        # No labels and no ticks.
        ax1.tick_params(labelbottom=False, labelleft=False, labelright=False,
                        labeltop=False,
                        length=0)
        ax1.imshow(img[i])
    plt.show()


start_number = input("Please Enter Start Number: ")
train_or_test = input("Please Enter train or test: ")

# The numbers to be recommended are as follows.
row_number = 5
column_number = 10
show_label_image(int(start_number), row_number, column_number, train_or_test)




