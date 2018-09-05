# coding: utf-8
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle

from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print('x_train size: {}'.format(len(x_train)))
print('x_test size: {}'.format(len(x_test)))
original_image_size = 28

def make_kl_transformation_matrix(data_type):
    """
    Make image eigen vectors from training data or test data.
    :param data_type: train or test
    :return: image eigen vectors
    """
    if data_type != 'train' and data_type != 'test':
        print("usage: make_kl_transformation_matrix(data_type)")
        print("data_type must be train or test. ({}).".format(data_type))
        return

    data_number = eval('len(x_{})'.format(data_type))

    img_matrix = eval('x_{}[0]'.format(data_type))

    # Check if the cov eigen dictionaries already exist.
    cov_eigen_values_dict_file_name = 'x_{}_cov_eigen_values_dict.pickle'.format(data_type)
    cov_eigen_vector_dict_file_name = 'x_{}_cov_eigen_vectors_dict.pickle'.format(data_type)
    if os.path.exists(cov_eigen_values_dict_file_name) \
            and os.path.exists(cov_eigen_vector_dict_file_name):
        with open(cov_eigen_values_dict_file_name, 'rb') as f:
            cov_eigen_values_dict = pickle.load(f)
        with open(cov_eigen_vector_dict_file_name, 'rb') as f:
            cov_eigen_vectors_dict = pickle.load(f)
    else:
        # Read data and add to img_matrix.
        for i in range(1, data_number):
        # for i in range(1, 5000):  # for debug
            #if eval('t_{}[i]'.format(data_type)) == 1:  # for debug. only '1'
            img = eval('x_{}[i]'.format(data_type))
            img_matrix = np.vstack((img_matrix, img))

        # Check img_matrix
        # for i in range(784):
        #     print('mean[{}]={}'.format(i, np.mean(img_matrix[:,i])))
        #     print('var=[{}]={}'.format(i, np.var(img_matrix[:,i])))

        # Calculate eigen values and eigen vectors.
        cov = np.cov(img_matrix, rowvar=False)
        w, v = np.linalg.eig(cov)
        cov_eigen_values_dict = np.real(w)
        cov_eigen_vectors_dict = np.real(v)

    # Save cov the eigen dictionaries.
    with open(cov_eigen_values_dict_file_name, 'wb') as f:
        pickle.dump(cov_eigen_values_dict, f)
    with open(cov_eigen_vector_dict_file_name, 'wb') as f:
        pickle.dump(cov_eigen_vectors_dict, f)

    return cov_eigen_values_dict, cov_eigen_vectors_dict


def calculate_weighted_L2_distance(x, y, w, dim):
    """
    Calculated
    :param x: Calculate the distance between x and y.
    :param y:
    :param w: weight vector. The length of the vector is equal to or greater than dim.
    :return: weighted distance
    """

    if len(x) != len(y):
        print("The size of x and y must be the same.")
        return
    if len(w) < dim:
        print("The length of the vector is equal to or greater than dim.")
        return

    distance = 0
    for i in range(dim):
        z = x[i] - y[i]
        distance += z * z / w[i]
    return distance


def calculate_L2_distance(x, y, dim):
    """
    Calculated
    :param x: Calculate the distance between x and y.
    :param y:
    :param dim: number of dimensions to be used
    :return: L2 distance
    """

    if len(x) != len(y):
        print("The size of x and y must be the same.")
        return

    distance = 0
    for i in range(dim):
        z = x[i] - y[i]
        distance += z * z
    return distance


def make_kl_expanded_dict(dict_type, expand_type):
    """
    Make expanded image dictionary from dict_type image dictionary
    by KL-expansion of expand_type eigen values matrix.
    :param dict_type: image dictionary source
    :param expand_type: eigen values matrix source
    :return: expanded dictionary
    """

    img_dict_file_name = 'x_{}_img_dict.pickle'.format(dict_type)
    eigen_vectors_file_name = 'x_{}_eigen_vectors.pickle'.format(expand_type)
    expanded_img_dict_file_name = 'x_{}_{}_expanded_img_dict.pickle'.format(
        dict_type, expand_type)
    expanded_img_dict = {}

    # Check if expanded dictionary already exist.
    if os.path.exists(expanded_img_dict_file_name):
        print('File already exists.')
        return

    # Load eigen values and eigen vectors.
    with open(img_dict_file_name, 'rb') as f:
        img_dict = pickle.load(f)
    with open(eigen_vectors_file_name, 'rb') as f:
        eigen_vectors_matrix = pickle.load(f)

    for c in range(cat_num):
        expanded_img_dict[c] = np.dot(img_dict[c], eigen_vectors_matrix)

    # Save expanded_img_dict
    with open(expanded_img_dict_file_name, 'wb') as f:
        pickle.dump(expanded_img_dict, f)

    return expanded_img_dict


def recognize_image_samples_weighted_distance \
                (data_type, dict_file, eigen_values_file,
                 eigen_vectors_matrix_file, dim, weighted=True):
    """
    Recognize image samples using KL-expanded image dictionary
    and weighted-L2 distance.
    :param data_type: train or test. samples to be recognized.
    :param dict_file:
    :param eigen_values_file: pickled file of eigen values of KL-expansion
    :param eigen_vectors_matrix_file: pickled file of eigen vectors matrix
        of KL-expansion
    :param dim: dimension to be used
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
    with open(dict_file, 'rb') as f:
        expanded_img_dict = pickle.load(f)
    with open(eigen_values_file, 'rb') as f:
        eigen_values = pickle.load(f)
    with open(eigen_vectors_matrix_file, 'rb') as f:
        eigen_vector_matrix = pickle.load(f)

    # Read data and KL-expand it,
    # and calculate weighted-L2 distance from the data
    # in the dictionary for each category.
    for i in range(data_number):
        img = eval('x_{}[i]'.format(data_type))
        img1 = np.dot(img, eigen_vector_matrix)
        label = eval('t_{}[i]'.format(data_type))
        for c in range(category_number):
            if weighted:
                distance_dict[c] = calculate_weighted_L2_distance(img1, expanded_img_dict[c], eigen_values, dim)
            else:
                distance_dict[c] = calculate_L2_distance(img1, expanded_img_dict[c], dim)
        sorted_distance = sorted(distance_dict.items(), key=lambda x: x[1])

        # if label != sorted_distance[0][0]:
        #     print('----sample {}------'.format(i))
        #     print("Wrong! answer={}".format(label))
        #     for k, v in sorted_distance:
        #         print(str(k) + ": " + str(v))
        #     print('----------')
        #     print(confusion_matrix)
        confusion_matrix[label, sorted_distance[0][0]] += 1

    return confusion_matrix


if __name__ == '__main__':
    dim = 200
    make_kl_expanded_dict('train', 'train')

    eigen_values_file = 'x_train_eigen_values.pickle'
    with open(eigen_values_file, 'rb') as f:
        eigen_values = pickle.load(f)
    total = np.sum(eigen_values)
    sub_total = 0

    x = [0] * len(eigen_values)
    y = [0] * len(eigen_values)
    for i in range(len(eigen_values)):
        sub_total += eigen_values[i]
        x[i] = i
        y[i] = sub_total/total*100
        print('{}: {} {} {}'.format(i, eigen_values[i], sub_total, sub_total / total * 100))

    plt.plot(x, y)
    plt.show()

    # Recognition!
    confusion_matrix = \
        recognize_image_samples_weighted_distance('test',
            'x_train_train_expanded_img_dict.pickle',
            'x_train_eigen_values.pickle',
            'x_train_eigen_vectors.pickle',
            dim,
            weighted=False)
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

"""
Weighted L2 distance
dim = 200
0:0.9295918367346939
1:0.9506607929515418
2:0.8013565891472868
3:0.8702970297029703
4:0.8839103869653768
5:0.7914798206278026
6:0.8883089770354906
7:0.8219844357976653
8:0.8213552361396304
9:0.8681863230921705
sum_total=10000.0
correct_total=8642.0
accuracy=0.8642

No-weighted L2 distance
dim = 200
0:0.8959183673469387
1:0.9621145374449339
2:0.7558139534883721
3:0.805940594059406
4:0.8268839103869654
5:0.6860986547085202
6:0.8632567849686847
7:0.8317120622568094
8:0.7371663244353183
9:0.8067393458870169
sum_total=10000.0
correct_total=8202.0
accuracy=0.8202
"""

