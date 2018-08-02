import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pylab import rcParams

from dataset.mnist import load_mnist

category_number = 10
# display image dictionary in row_number x column_number format
row_number = 5
column_number = 10
eigen_number = 5

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print('x_train size: {}'.format(len(x_train)))
print('x_test size: {}'.format(len(x_test)))
original_image_size = 28

# Extend the display size.
rcParams['figure.figsize'] = 10, 10
# rcParams['figure.figsize'] = 20, 20  # for ASUS Note PC


def calculate_multiple_similarity(x, eigen_values, eigen_vectors, dim):
    """
    :param x: Input vectors. The multiple similarity for x
    :param eigen_values: eigen values for a category
    :param eigen_vectors: eigen vectors for a category
    :param dim: dimension to be used
    :return: multiple similarity of x for a category
    """

    if len(eigen_values) < dim:
        print("The length of the eigen vector is equal to or greater than dim.")
        return

    similarity = 0
    for i in range(dim):
        work = cp.dot(eigen_vectors[:,i], x)
        # work = np.dot(eigen_vectors[:,i], x)
        similarity += eigen_values[i] * work * work

    return similarity / eigen_values[0]


def make_multiple_similarity_dict(data_type):
    """
    Make multiple similarity dictionary from dict_type image data
    :param dict_type: image data source type
    :return: eigen values dictionary, eigen vectors dictionary
    """

    if data_type != 'train' and data_type != 'test':
        print("usage: make_multiple_similarity_dict(data_type)")
        print("data_type must be train or test. ({}).".format(data_type))
        return

    eigen_values_dict_file_name = 'x_{}_eigen_values_dict.pickle'.format(data_type)
    eigen_vectors_dict_file_name = 'x_{}_eigen_vectors_dict.pickle'.format(data_type)

    category_number = 10
    # display eigen vectors dictionary in row_number x column_number format
    row_number = 5
    column_number = 10

    data_number = eval('len(x_{})'.format(data_type))

    fig = plt.figure()
    img_data = {}
    count_data = {}
    eigen_values_dict = {}
    eigen_vectors_dict = {}
    eigen_number = 5

    # Read data and add to each category.
    # for i in range(data_number):
    for i in range(1000):  # for debug
        img = eval('x_{}[i]'.format(data_type))
        label = eval('t_{}[i]'.format(data_type))
        if label in img_data:
            img_data[label] = np.vstack((img_data[label], img))
            count_data[label] += 1
        else:
            img_data[label] = img
            count_data[label] = 1

    # Get eigen values w[] and eigen vectors v[][]
    for c in range(category_number):
        # autocorrelation_matrix = np.dot(img_data[c].T, img_data[c]) / count_data[c]
        autocorrelation_matrix = cp.dot(img_data[c].T, img_data[c]) / count_data[c]
        w, v = np.linalg.eig(autocorrelation_matrix)
        eigen_values_dict[c] = np.real(w)
        eigen_vectors_dict[c] = np.real(v)
        print('c={}'.format(c))
        print('w={}'.format(eigen_values_dict[c][:10]))
        print('v={}'.format(eigen_vectors_dict[c][:,0]))

    # Display eigen vectors of image for each category.
    for c in range(category_number):
        img_c = eigen_vectors_dict[c]
        for i in range(eigen_number):
            ax1 = fig.add_subplot(row_number, column_number, c * eigen_number + i + 1)
            img = img_c[:,i].reshape(original_image_size, original_image_size)
            ax1.set_title('{} {} ({})'.format(data_type, c, count_data[c]))

            # No labels and no ticks.
            ax1.tick_params(labelbottom=False, labelleft=False, labelright=False,
                            labeltop=False,
                            length=0)
            ax1.imshow(img, cmap='bone')
    plt.show()

    # Save the average image dictionary.
    with open(eigen_values_dict_file_name, 'wb') as f:
        pickle.dump(eigen_values_dict, f)
    with open(eigen_vectors_dict_file_name, 'wb') as f:
        pickle.dump(eigen_vectors_dict, f)

    return eigen_values_dict, eigen_vectors_dict

def recognize_image_samples_multiple_similarity \
            (data_type, eigen_values_file, eigen_vectors_matrix_file, dim):
    """
    Recognize image samples using multiple similarity
    :param data_type: train or test. samples to be recognized.
    :param eigen_values_file: pickled file of eigen values
        of autocorrelation matrix
    :param eigen_vectors_matrix_file: pickled file of eigen vectors matrix
        of autocorrelation matrix
    :param dim: dimension to be used
    :return: confusion matrix
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

    # Load eigen values and eigen vectors file.
    with open(eigen_values_file, 'rb') as f:
        eigen_values_dict = pickle.load(f)
    with open(eigen_vectors_matrix_file, 'rb') as f:
        eigen_vectors_dict = pickle.load(f)

    # Read data and calculate multiple similarity to the data
    # for each category.
    for i in range(data_number):
        img = eval('x_{}[i]'.format(data_type))
        label = eval('t_{}[i]'.format(data_type))
        for c in range(category_number):
            distance_dict[c] = \
                calculate_multiple_similarity(img, eigen_values_dict[c], eigen_vectors_dict[c], dim)
        sorted_distance = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)

        # if label != sorted_distance[0][0]:
        #     print('----sample {}------'.format(i))
        #     print("Wrong! answer={}".format(label))
        #     for k, v in sorted_distance:
        #         print(str(k) + ": " + str(v))
        #     print('----------')
        # #     print(confusion_matrix)
        confusion_matrix[label, sorted_distance[0][0]] += 1

    return confusion_matrix


if __name__ == '__main__':

    make_multiple_similarity_dict('train')
    eigen_values_file = 'x_train_eigen_values_dict.pickle'
    eigen_vectors_file = 'x_train_eigen_vectors_dict.pickle'
    dim = 600
    confusion_matrix = recognize_image_samples_multiple_similarity \
        ('train', eigen_values_file, eigen_vectors_file, dim)

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

