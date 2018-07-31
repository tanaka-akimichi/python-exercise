import numpy as np
import pickle

cat_num = 10

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
    if len(w) >= dim:
        print("The length of the vector is equal to or greater than dim.")
        return

    distance = 0
    for i in range(dim):
        z = x[i] = y[i]
        distance += z * z * w[i]
    z = x - y
    return distance


def make_expanded_dict(dict_type, expand_type):
    """
    Make expanded image dictionary from dict_type image dictionary
    by KL-expansion of expand_type eigen values matrix.
    :param dict_type: image dictionary source
    :param expand_type: eigen values matrix source
    :return: expanded dictionary
    """

    img_dict_file_name = 'x_{}_img_dict.pickle'.format(dict_type)
    eigen_vectors_file_name = 'x_{}_eigen_vectors.pickle'.format(expand_type)
    expanded_img_dict_file_name = 'x_{}_{}_expanded_img_dict.pickle'.format(dict_type, expand_type)
    expanded_img_dict = {}

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

if __name__ == '__main__':
    make_expanded_dict('train', 'train')