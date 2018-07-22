import numpy as np
import time

# Number of dimensions of vector and matrix
num = 10000

# Repetition number of time measurement
repeat = 1000

# Generate an num-dimensional row vector whose elements are
# random number from -1.0 to 1.0.
row_x = np.matrix([np.random.rand(num) * 2.0 - 1.0])

# Convert to column vector.
column_x = row_x.T

print("row_x:\n{}".format(row_x))
print("dim of row_x={}".format(row_x.ndim))

print("column_x:\n{}".format(column_x))
print("dim of column_x={}".format(column_x.ndim))

# Generate an num x num-dimensional matrix whose elements are
# random number from -1.0 to 1.0.
a = np.matrix([np.random.rand(num * num) * 2.0 - 1.0]).reshape(num, num)
print("matrix_a:\n{}".format(a))
print("dim of matrix={}".format(a.ndim))

# Compare row_x . a and aT . column_x
y = np.dot(row_x, a)
y1 = np.dot(a.T, column_x)
print("y:\n{}".format(y))
print("y1:\n{}".format(y1))

# Measure processing time.
# row_vector . matrix
start = time.time()
for i in range(0, repeat):
    y = np.dot(row_x, a)
elapsed_time = time.time() - start
print("elapsed time = {}".format(elapsed_time))

# matrix . column_vector
start = time.time()
for i in range(0, repeat):
    y1 = np.dot(a, column_x)
elapsed_time = time.time() - start
print("elapsed time = {}".format(elapsed_time))

