import numpy as np
import timeit

# Number of dimensions of vector and matrix
num = 1000

# Repetition number of time measurement
repeat = 100000

# Generate an num-dimensional row vector whose elements are
# random number from -1.0 to 1.0.
row_x = np.array([np.random.rand(num) * 2.0 - 1.0])

# Generate an num-dimensional row vector whose elements are
# random number from -1.0 to 1.0.
row_x_matrix = np.matrix([np.random.rand(num) * 2.0 - 1.0])

# Convert to column vector.
column_x = row_x_matrix.T

# Generate an num x num-dimensional matrix whose elements are
# random number from -1.0 to 1.0.
a = np.matrix([np.random.rand(num * num) * 2.0 - 1.0]).reshape(num, num)

at = a.T

print("row_x:\n{}".format(row_x))
print("dim of row_x={}".format(row_x.ndim))

print("column_x:\n{}".format(column_x))
print("dim of column_x={}".format(column_x.ndim))

print("matrix_a:\n{}".format(a))
print("dim of matrix={}".format(a.ndim))

def row_vector_matrix(row_vector, mat):
    y = np.dot(row_vector, mat)


def matrix_column_vector(mat, column_vector):
    y = np.dot(mat, column_vector)

# Measure processing time.
result1 = timeit.timeit('row_vector_matrix(row_x, a)',
                        globals=globals(), number=repeat)
result2 = timeit.timeit('matrix_column_vector(at, column_x)',
                        globals=globals(), number=repeat)

print("elapsed time(row_vector x matrix) = {}".format(result1))
print("elapsed time(matrix x column_vector) = {}".format(result2))
