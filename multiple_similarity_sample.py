import matplotlib.pyplot as plt
import numpy as np


"""
Set mean values and a covariance matrix, 
and generate sample points. 
"""
feature_vectors_dimension = 3
sample_number = 1000

mean = np.array([0.0, 1.0, 2.0])
cov = np.array([
    [1.0, 0.2, 0.5],
    [0.2, 1.0, 0.7],
    [0.5, 0.7, 1.0]])

data_matrix = np.random.multivariate_normal(mean, cov, 1)

for i in range(1, sample_number):
    data = np.random.multivariate_normal(mean, cov, 1)
    data_matrix = np.vstack((data_matrix, data))


"""
Calculate a autocorrelation matrix from the generated sample
points, and its eigen values and eigen vectors.
"""

# Get eigen values w[] and eigen vectors v[][]
autocorrelation_matrix = np.dot(data_matrix.T, data_matrix) / sample_number
w, v = np.linalg.eig(autocorrelation_matrix)
print('autocorrelation matrix:\n{}'.format(autocorrelation_matrix))
print('w = {}'.format(w))
print('v = {}'.format(v))
print('vT = {}'.format(v.T))

# eigen vectors corresponding to w[0], w[1] and w[2]
print('v0 = {}'.format(v[:,0]))
print('v1 = {}'.format(v[:,1]))
print('v2 = {}'.format(v[:,2]))

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot(data_matrix[:,0], data_matrix[:,1], 'x')
plt.title("covariance=0.2")
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
ax1 = fig.add_subplot(222)
ax1.plot(data_matrix[:,1], data_matrix[:,2], 'x')
plt.title("covariance=0.7")
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
ax1 = fig.add_subplot(223)
ax1.plot(data_matrix[:,2], data_matrix[:,0], 'x')
plt.title("covariance=0.5")
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)

plt.show()

transformed_data_matrix = np.dot(data_matrix[0], v)
for i in range(1, sample_number):
    data = np.dot(data_matrix[i], v)
    transformed_data_matrix = np.vstack((transformed_data_matrix, data))

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot(transformed_data_matrix[:,0], transformed_data_matrix[:,1], 'x')
#plt.title("covariance=0.2")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
ax1 = fig.add_subplot(222)
ax1.plot(transformed_data_matrix[:,1], transformed_data_matrix[:,2], 'x')
#plt.title("covariance=0.7")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
ax1 = fig.add_subplot(223)
ax1.plot(transformed_data_matrix[:,2], transformed_data_matrix[:,0], 'x')
#plt.title("covariance=0.5")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()





