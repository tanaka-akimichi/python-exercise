import numpy as np
import matplotlib.pyplot as plt
import os
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


def calculate_subspace_similarity(x, eigen_vectors, dim):
    """
    :param x: Input vectors. The multiple similarity for x
    :param eigen_vectors: eigen vectors for a category
    :param dim: dimension to be used
    :return: subspace similarity of x for a category
    """

    similarity = 0
    for i in range(dim):
        work = np.dot(eigen_vectors[:,i], x)
        similarity += work * work

    return similarity


def make_subspace_similarity_dict(data_type):
    """
    Make multiple similarity dictionary from dict_type image data
    :param dict_type: image data source type
    :return: eigen values dictionary, eigen vectors dictionary
    """

    if data_type != 'train' and data_type != 'test':
        print("usage: make_multiple_similarity_dict(data_type)")
        print("data_type must be train or test. ({}).".format(data_type))
        return

    category_number = 10
    # display eigen vectors dictionary in row_number x column_number format
    row_number = 5
    column_number = 10

    data_number = eval('len(x_{})'.format(data_type))

    fig = plt.figure()
    img_data = {}
    count_data = {}
    subspace_eigen_values_dict = {}
    subspace_eigen_vectors_dict = {}
    eigen_number = 5  # for display
    
    # Check if the subspace eigen dictionaries already exist.
    subspace_eigen_values_dict_file_name = 'x_{}_subspace_eigen_values_dict.pickle'.format(data_type)
    subspace_eigen_vectors_dict_file_name = 'x_{}_subspace_eigen_vectors_dict.pickle'.format(data_type)
    if os.path.exists(subspace_eigen_values_dict_file_name) \
            and os.path.exists(subspace_eigen_vectors_dict_file_name):
        with open(subspace_eigen_values_dict_file_name, 'rb') as f:
            subspace_eigen_values_dict = pickle.load(f)
        with open(subspace_eigen_vectors_dict_file_name, 'rb') as f:
            subspace_eigen_vectors_dict = pickle.load(f)
    else:
        # Read data and add to each category.
        for i in range(data_number):
        # for i in range(1000):  # for debug
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
            autocorrelation_matrix = np.dot(img_data[c].T, img_data[c]) / count_data[c]
            w, v = np.linalg.eigh(autocorrelation_matrix)
            subspace_eigen_values_dict[c] = w[::-1]  # reverse
            subspace_eigen_vectors_dict[c] = v
            for i in range(v.shape[0]):
                # Reverse with respect to each row
                subspace_eigen_vectors_dict[c][i] = subspace_eigen_vectors_dict[c][i][::-1]
            print('c={}'.format(c))
            print('w={}'.format(subspace_eigen_values_dict[c][:10]))
            print('v={}'.format(subspace_eigen_vectors_dict[c][:,0]))

    # Display eigen vectors of image for each category.
    for c in range(category_number):
        img_c = subspace_eigen_vectors_dict[c]
        for i in range(eigen_number):
            ax1 = fig.add_subplot(row_number, column_number, c * eigen_number + i + 1)
            img = img_c[:,i].reshape(original_image_size, original_image_size)
            ax1.set_title('{} {}'.format(data_type, c))

            # No labels and no ticks.
            ax1.tick_params(labelbottom=False, labelleft=False, labelright=False,
                            labeltop=False,
                            length=0)
            ax1.imshow(img, cmap='bone')
    plt.show()

    # Save the subspace eigen values and vectors dictionaries.
    with open(subspace_eigen_values_dict_file_name, 'wb') as f:
        pickle.dump(subspace_eigen_values_dict, f)
    with open(subspace_eigen_vectors_dict_file_name, 'wb') as f:
        pickle.dump(subspace_eigen_vectors_dict, f)

    return subspace_eigen_values_dict, subspace_eigen_vectors_dict


def recognize_image_samples_subspace_similarity \
            (data_type, eigen_vectors_matrix_file, dim):
    """
    Recognize image samples using similarity in subspaces.
    :param data_type: train or test. samples to be recognized.
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
    similarity_dict = {}

    """
    Elements of confusion_matrix[i, j] represents the number 
    where real category is "i" and predicted category is "j."
    """
    confusion_matrix = np.zeros((10, 10))

    # Load eigen vectors file.
    with open(eigen_vectors_matrix_file, 'rb') as f:
        eigen_vectors_dict = pickle.load(f)

    # Read data and calculate subspace similarity to the data
    # for each category.
    for i in range(data_number):
        img = eval('x_{}[i]'.format(data_type))
        label = eval('t_{}[i]'.format(data_type))
        for c in range(category_number):
            similarity_dict[c] = \
                calculate_subspace_similarity(img, eigen_vectors_dict[c], dim)
        sorted_distance = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

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

    make_subspace_similarity_dict('train')
    # eigen_values_file = 'x_train_subspace_eigen_values_dict.pickle'
    eigen_vectors_file = 'x_train_subspace_eigen_vectors_dict.pickle'
    dim = 30
    confusion_matrix = recognize_image_samples_subspace_similarity \
        ('test', eigen_vectors_file, dim)

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

    print('dim={}'.format(dim))
    print('sum_total={}'.format(sum_total))
    print('correct_total={}'.format(correct_total))
    print('accuracy={}'.format(correct_total / sum_total))

"""
The best accuracy was obtained when the dimension used is 30.
dim=10
accuracy=0.9485

dim=20
accuracy=0.9573

dim=30
accuracy=0.9574 *****

dim=40
accuracy=0.957

dim=50
accuracy=0.9514

dim=60
accuracy=0.9452
"""