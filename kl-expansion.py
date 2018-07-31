# coding: utf-8
import matplotlib.pyplot as plt
# import sys, os
# sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
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

    category_number = 10
    # display image dictionary in row_number x column_number format
    row_number = 5
    column_number = 10
    eigen_number = 5

    data_number = eval('len(x_{})'.format(data_type))

    img_matrix = eval('x_{}[0]'.format(data_type))

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
    eigen_values_vector = np.real(w)
    eigen_vectors_matrix = np.real(v)

    return eigen_values_vector, eigen_vectors_matrix

if __name__ == '__main__':
    # eigen_values_vector, eigen_vectors_matrix = make_kl_transformation_matrix('train')

    # Save eigen values and eigen vectors.
    eigen_values_file_name = 'x_train_eigen_values.pickle'
    eigen_vectors_file_name = 'x_train_eigen_vectors.pickle'
    # with open(eigen_values_file_name, 'wb') as f:
    #     pickle.dump(eigen_values_vector, f)
    # with open(eigen_vectors_file_name, 'wb') as f:
    #     pickle.dump(eigen_vectors_matrix, f)


    # eigen_values_vector, eigen_vectors_matrix = make_kl_transformation_matrix('test')

    # Save eigen values and eigen vectors.
    eigen_values_file_name = 'x_train_eigen_values.pickle'
    eigen_vectors_file_name = 'x_train_eigen_vectors.pickle'
    # with open(eigen_values_file_name, 'wb') as f:
    #     pickle.dump(eigen_values_vector, f)
    # with open(eigen_vectors_file_name, 'wb') as f:
    #     pickle.dump(eigen_vectors_matrix, f)

    # Load eigen values and eigen vectors.
    with open(eigen_values_file_name, 'rb') as f:
       eigen_values_vector = pickle.load(f)
    with open(eigen_vectors_file_name, 'rb') as f:
       eigen_vectors_matrix = pickle.load(f)

    # print('w = {}'.format(eigen_values_vector))
    # print('v = {}'.format(eigen_vectors_matrix))
    # print('vT = {}'.format(eigen_vectors_matrix.T))
    #
    # # eigen vectors corresponding to w[0], w[1] and w[2]
    # print('v0 = {}'.format(eigen_vectors_matrix[:, 0]))
    # print('v1 = {}'.format(eigen_vectors_matrix[:, 1]))
    # print('v2 = {}'.format(eigen_vectors_matrix[:, 2]))

    # Initialize
    x = x_train[0]
    x1 = np.dot(x, eigen_vectors_matrix)

    for i in range(1, 60000):
        x = x_train[i]
        x1 = np.vstack((x1, np.dot(x, eigen_vectors_matrix)))

    print(eigen_values_vector[:10])

    # Check the values!
    for i in range(10):
        mean = np.mean(x1[:,i])
        var = np.var(x1[:,i])
        print('mean[{}]={}'.format(i, mean))
        print('var[{}]={}'.format(i, var))  # var[i] should be eigen value[i]

    plt.plot(x1[:,0], x1[:,1], 'x')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.show()

    plt.plot(x1[:,2], x1[:,3], 'x')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.show()

    plt.plot(x1[:,4], x1[:,5], 'x')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.show()
