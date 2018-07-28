# coding: utf-8
import matplotlib.pyplot as plt
# import sys, os
# sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from PIL import Image
from pylab import rcParams

from dataset.mnist import load_mnist


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def calculate_L2_distance(x, y):
    dim_x = len(x)
    dim_y = len(y)
    if dim_x != dim_y:
        print("The size of x and y must be the same.")
        return

    z = x - y
    return np.linalg.norm(z)

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print('x_train size: {}'.format(len(x_train)))
print('x_test size: {}'.format(len(x_test)))
original_image_size = 28

# Extend the display size.
rcParams['figure.figsize'] = 10, 10
# rcParams['figure.figsize'] = 20, 20  # for ASUS Note PC


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
        img[i] = eval('x_{}[start + i]'.format(data_type))
        label[i] = eval('t_{}[start + i]'.format(data_type))
        ax1.set_title('{} {} ({})'.format(data_type, start + i, str(label[i])))

        img[i] = img[i].reshape(original_image_size, original_image_size)

        # No labels and no ticks.
        ax1.tick_params(labelbottom=False, labelleft=False, labelright=False,
                        labeltop=False,
                        length=0)
        ax1.imshow(img[i])
    plt.show()


def make_image_dictionary(data_type):
    """
    Make image dictionary from training data or test data.
    :param data_type: train or test
    :return: image dictionary
    """
    if data_type != 'train' and data_type != 'test':
        print("usage: make_image_discionary(data_type)")
        print("data_type must be train or test. ({}).".format(data_type))
        return

    category_number = 10
    # display image dictionary in row_number x column_number format
    row_number = 2
    column_number = 5

    data_number = eval('len(x_{})'.format(data_type))

    fig = plt.figure()
    img_dict = {}
    count_dict = {}

    # Read data and add for each category.
    for i in range(data_number):
        img = eval('x_{}[i]'.format(data_type))
        label = eval('t_{}[i]'.format(data_type))
        if label in img_dict:
            img_dict[label] += img
            count_dict[label] += 1
        else:
            img_dict[label] = img
            count_dict[label] = 1

    # Calculate the average
    for c in range(category_number):
        img_dict[c] /= count_dict[c]

    # Load the dictionary file.
    # with open('x_train_img_dict.pickle', 'rb') as f:
    #    img_dict = pickle.load(f)

    # Display the average image for each category.
    for c in range(category_number):
        ax1 = fig.add_subplot(row_number, column_number, c+1)
        img = img_dict[c].reshape(original_image_size, original_image_size)
        ax1.set_title('{} {} ({})'.format(data_type, c, count_dict[c]))

        # No labels and no ticks.
        ax1.tick_params(labelbottom=False, labelleft=False, labelright=False,
                        labeltop=False,
                        length=0)
        ax1.imshow(img)

    # Save the average image dictionary.
    dict_name = 'x_{}_img_dict.pickle'.format(data_type)
    with open(dict_name, 'wb') as f:
        pickle.dump(img_dict, f)

    plt.show()

    return img_dict, count_dict


def make_image_eigen_vectors(data_type):
    """
    Make image eigen vectors from training data or test data.
    :param data_type: train or test
    :return: image eigen vectors
    """
    if data_type != 'train' and data_type != 'test':
        print("usage: make_image_eigen_vectors(data_type)")
        print("data_type must be train or test. ({}).".format(data_type))
        return

    category_number = 10
    # display image dictionary in row_number x column_number format
    row_number = 5
    column_number = 10
    eigen_number = 5

    data_number = eval('len(x_{})'.format(data_type))

    fig = plt.figure()
    img_dict = {}
    count_dict = {}
    eigen_values_dict = {}
    eigen_vectors_dict = {}

    # Read data and add for each category.
    # for i in range(data_number):
    for i in range(2000):  # for debug
        img = eval('x_{}[i]'.format(data_type))
        label = eval('t_{}[i]'.format(data_type))
        if label in img_dict:
            img_dict[label] = np.vstack((img_dict[label], img))
            count_dict[label] += 1
        else:
            img_dict[label] = img
            count_dict[label] = 1

    # Calculate eigen values and eigen vectors.
    for c in range(category_number):
        cov = np.cov(img_dict[c], rowvar=False)
        # Check if cov is symmetric
        w, v = np.linalg.eig(cov)

        # Take a real part because eigen values may be complex.
        eigen_values_dict[c] = np.real(w)
        eigen_vectors_dict[c] = np.real(v)

    # Display eigen vectors of image for each category.
    for c in range(category_number):
        img_c = eigen_vectors_dict[c]
        for i in range(eigen_number):
            ax1 = fig.add_subplot(row_number, column_number, c * eigen_number + i + 1)
            img = img_c[:,i].reshape(original_image_size, original_image_size)
            ax1.set_title('{} {} ({})'.format(data_type, c, count_dict[c]))

            # No labels and no ticks.
            ax1.tick_params(labelbottom=False, labelleft=False, labelright=False,
                            labeltop=False,
                            length=0)
            ax1.imshow(img)
    plt.show()

    return eigen_values_dict, eigen_vectors_dict


def recognize_image_samples(data_type):
    """
    Recognize image samples using image dictionary made from train data
    and L2 distance.
    :param data_type: train or test. samples to be recognized.
    :return: image dictionary
    """
    if data_type != 'train' and data_type != 'test':
        print("usage: recognize_image_samples(data_type)")
        print("data_type must be train or test. ({}).".format(data_type))
        return

    category_number = 10
    data_number = eval('len(x_{})'.format(data_type))
    distance_dict = {}

    """
    Elements of confusion_matrix[i, j] represents the number 
    where real category is "i" and predicted category is "j."
    """
    confusion_matrix = np.zeros((10, 10))

    # Load the dictionary file.
    with open('x_train_img_dict.pickle', 'rb') as f:
        img_dict = pickle.load(f)

    # Read data and calculate L2 distance from the data
    # in the dictionary for each category.
    for i in range(data_number):
        img = eval('x_{}[i]'.format(data_type))
        label = eval('t_{}[i]'.format(data_type))
        for c in range(category_number):
            distance_dict[c] = calculate_L2_distance(img, img_dict[c])
        sorted_distance = sorted(distance_dict.items(), key=lambda x: x[1])

        # if label != sorted_distance[0][0]:
        #     print('----sample {}------'.format(i))p
        #     print("Wrong!")
        #     for k, v in sorted_distance:
        #         print(str(k) + ": " + str(v))
        #     print('----------')
        #     print(confusion_matrix)
        confusion_matrix[label, sorted_distance[0][0]] += 1

    return confusion_matrix

# start_number = input("Please Enter Start Number: ")
# train_or_test = input("Please Enter train or test: ")

# The numbers to be recommended are as follows.
row_number = 5
column_number = 10
# show_label_image(int(start_number), row_number, column_number, train_or_test)
# make_image_dictionary('train')
# make_image_dictionary('test')
v, w = make_image_eigen_vectors('train')
print('eigen values={}'.format(v[0]))
confusion_matrix = recognize_image_samples('train')
print(confusion_matrix)

recall = np.empty(10)

sum_total = 0
correct_total = 0
for i in range(10):
    sum = 0
    for j in range(10):
        sum += confusion_matrix[i, j]
        sum_total += confusion_matrix[i, j]
    recall[i] = confusion_matrix[i, i] / sum
    correct_total += confusion_matrix[i, i]

for i in range(10):
    print('{}:{}'.format(i, recall[i]))

print('sum_total={}'.format(sum_total))
print('correct_total={}'.format(correct_total))
print('accuracy={}'.format(correct_total / sum_total))

