# coding: utf-8
import matplotlib.pyplot as plt
import sys, os
import numpy as np
import pickle
from PIL import Image
from pylab import rcParams

from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print('x_train size: {}'.format(len(x_train)))
print('x_test size: {}'.format(len(x_test)))
original_image_size = (28, 28)

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
    # if data_type != 'train' and data_type != 'test':
    #     print("usage: show_label_image(start, row_number, column_number, "
    #           "data_type)")
    #     print("data_type must be train or test. ({}).".format(data_type))
    #     return

    fig = plt.figure()

    img = [0] * (row_number * column_number)
    label = [0] * (row_number * column_number)

    for i in range(row_number * column_number):
        ax1 = fig.add_subplot(row_number, column_number, i+1)
        img[i] = eval('x_{}[start + i]'.format(data_type))
        label[i] = eval('t_{}[start + i]'.format(data_type))
        ax1.set_title('{} {} ({})'.format(data_type, start + i, str(label[i])))

        img[i] = img[i].reshape(original_image_size)

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
    # if data_type != 'train' and data_type != 'test':
    #     print("usage: make_image_discionary(data_type)")
    #     print("data_type must be train or test. ({}).".format(data_type))
    #     return

    category_number = 10
    # display image dictionary in row_number x column_number format
    row_number = 2
    column_number = 5

    data_number = eval('len(x_{})'.format(data_type))

    fig = plt.figure()
    img_dict = {}
    count_dict = {}

    # Check if the dictionaries already exist.
    img_dict_file_name = 'x_{}_img_dict.pickle'.format(data_type)
    count_dict_file_name = 'x_{}_count_dict.pickle'.format(data_type)
    if os.path.exists(img_dict_file_name) and os.path.exists(count_dict_file_name):
        with open(img_dict_file_name, 'rb') as f:
            img_dict = pickle.load(f)
        with open(count_dict_file_name, 'rb') as f:
            count_dict = pickle.load(f)
    else:
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

    # Display the average image for each category.
    for c in range(category_number):
        ax1 = fig.add_subplot(row_number, column_number, c+1)
        img = img_dict[c].reshape(original_image_size)
        ax1.set_title('{} {} ({})'.format(data_type, c, count_dict[c]))

        # No labels and no ticks.
        ax1.tick_params(labelbottom=False, labelleft=False, labelright=False,
                        labeltop=False,
                        length=0)
        ax1.imshow(img)

    # Save the average image dictionary.
    with open(img_dict_file_name, 'wb') as f:
        pickle.dump(img_dict, f)
    with open(count_dict_file_name, 'wb') as f:
        pickle.dump(count_dict, f)

    plt.show()

    return img_dict, count_dict


def make_image_eigen_vectors(data_type):
    """
    Make image eigen vectors from training data or test data.
    :param data_type: train or test
    :return: image eigen vectors
    """
    # if data_type != 'train' and data_type != 'test':
    #     print("usage: make_image_eigen_vectors(data_type)")
    #     print("data_type must be train or test. ({}).".format(data_type))
    #     return

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

    # Check if the eigen dictionaries already exist.
    eigen_values_dict_file_name = 'x_{}_eigen_values_dict.pickle'.format(data_type)
    eigen_vector_dict_file_name = 'x_{}_eigen_vectors_dict.pickle'.format(data_type)
    if os.path.exists(eigen_values_dict_file_name) and os.path.exists(eigen_vector_dict_file_name):
        with open(eigen_values_dict_file_name, 'rb') as f:
            eigen_values_dict = pickle.load(f)
        with open(eigen_vector_dict_file_name, 'rb') as f:
            eigen_vectors_dict = pickle.load(f)
    else:
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
            w, v = np.linalg.eig(cov)

            # Take a real part because eigen values may be complex.
            eigen_values_dict[c] = np.real(w)
            eigen_vectors_dict[c] = np.real(v)

    # Display eigen vectors of image for each category.
    for c in range(category_number):
        img_c = eigen_vectors_dict[c]
        for i in range(eigen_number):
            ax1 = fig.add_subplot(row_number, column_number, c * eigen_number + i + 1)
            img = img_c[:,i].reshape(original_image_size)
            ax1.set_title('{} {}'.format(data_type, c))

            # No labels and no ticks.
            ax1.tick_params(labelbottom=False, labelleft=False, labelright=False,
                            labeltop=False,
                            length=0)
            ax1.imshow(img)
    plt.show()

    return eigen_values_dict, eigen_vectors_dict


if __name__ == '__main__':

    start_number = input("Please Enter Start Number(start with 0): ")
    try:
        int_start_num = int(start_number)
        # print("num={}".format(int_start_num))
    except ValueError:
        print("Start number must be an integer.")

    train_or_test = input("Please Enter train or test: ")

    if train_or_test == 'train':
        if int_start_num < 0 or int_start_num > 59950:
            print('Start number must be from 0 to 59950 for train data.')
            exit()
    elif train_or_test == 'test':
        if int_start_num < 0 or int_start_num > 9950:
            print('Start number must be from 0 to 9950 for test data.')
            exit()
    else:
        print("data_type must be train or test. ({}).".format(train_or_test))
        exit()

    # The numbers to be recommended are as follows.
    row_number = 5
    column_number = 10

    show_label_image(int(start_number), row_number, column_number, train_or_test)
    make_image_dictionary(train_or_test)
    # make_image_dictionary('test')

    v, w = make_image_eigen_vectors(train_or_test)

    sum_all = np.sum(v[0])
    sum = 0
    print('i value accumulation ratio(%)')
    for i in range(len(v[0])):
        sum += v[0][i]
        print('{} {} {} {}'.format(i, v[0][i], sum, sum / sum_all * 100))

    """
    i value accumulation ratio(%)
    0 72.48078155517578 72.48078155517578 61.2352671064512
    1 8.721412658691406 81.20219421386719 68.60353800863514
    2 5.826434135437012 87.0286283493042 73.52599114595446
    3 3.4116363525390625 90.44026470184326 76.40830641402002
    4 2.145717144012451 92.58598184585571 78.22111195542372
    5 1.7848519086837769 94.37083375453949 79.72904110613949
    6 1.5841861963272095 95.9550199508667 81.06743816529098
    7 1.402768611907959 97.35778856277466 82.25256488158253
    8 1.1821420192718506 98.53993058204651 83.25129558998093
    9 1.1121227741241455 99.65205335617065 84.19087065619098
    10 0.9939224720001221 100.64597582817078 85.03058439478663
    11 0.8569395542144775 101.50291538238525 85.75456834433139
    12 0.8177820444107056 102.32069742679596 86.44547013719117
    13 0.7357387542724609 103.05643618106842 87.06705779355694
    14 0.6464670300483704 103.70290321111679 87.61322438297749
    15 0.5708078742027283 104.27371108531952 88.09547046108665
    16 0.5557574033737183 104.82946848869324 88.56500117408083
    17 0.5138680338859558 105.3433365225792 88.9991417233038
    18 0.5034317970275879 105.84676831960678 89.42446523526671
    19 0.4220869243144989 106.26885524392128 89.7810646675295
    20 0.406838983297348 106.67569422721863 90.12478190231906
    21 0.37136051058769226 107.04705473780632 90.4385252087611
    22 0.3372286558151245 107.38428339362144 90.72343227476823
    23 0.3122577965259552 107.6965411901474 90.98724275205707
    24 0.29751917719841003 107.99406036734581 91.23860133145052
    25 0.28413301706314087 108.27819338440895 91.47865063583583
    26 0.2745059132575989 108.55269929766655 91.71056649766788
    27 0.26863235235214233 108.82133165001869 91.93752009138977
    28 0.250230610370636 109.07156226038933 92.14892700416694
    29 0.24676620960235596 109.31832846999168 92.35740702374763
    30 0.22433115541934967 109.54265962541103 92.54693282531402
    31 0.21066179871559143 109.75332142412663 92.72491009372487
    32 0.1991225630044937 109.95244398713112 92.89313845813923
    33 0.19852757453918457 110.1509715616703 93.06086414754351
    34 0.18954767286777496 110.34051923453808 93.22100318203542
    35 0.181675523519516 110.5221947580576 93.37449144431295
    36 0.1748805195093155 110.69707527756691 93.52223895881396
    37 0.16611726582050323 110.86319254338741 93.66258285308845
    38 0.16464510560035706 111.02783764898777 93.80168299526245
    39 0.14844894409179688 111.17628659307957 93.92709983746651
    40 0.1429378092288971 111.31922440230846 94.04786060659646
    41 0.13840149343013763 111.4576258957386 94.1647888769087
    42 0.13072100281715393 111.58834689855576 94.27522829756862
    43 0.1251533031463623 111.71350020170212 94.38096385647106
    44 0.12041199952363968 111.83391220122576 94.48269373293546
    45 0.11723007261753082 111.95114227384329 94.58173536377355
    46 0.11491469293832779 112.06605696678162 94.67882084996104
    47 0.10967147350311279 112.17572844028473 94.77147661097554
    48 0.10798219591379166 112.28371063619852 94.86270518864326
    49 0.10512110590934753 112.38883174210787 94.95151657920454
    50 0.10263761878013611 112.491469360888 95.0382297998172
    """

