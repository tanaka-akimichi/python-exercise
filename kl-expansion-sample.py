import matplotlib.pyplot as plt
import numpy as np


"""
How to use np.cov().
For example, let the first row of matrix b represent the score of
each person's mathematics.
Likewise, the second row represents the Japanese score, 
and the third row represents the English Score.
For the variance, invariant variance is calculated.
"""
# Data for Mathematics, Japanese and English
b = np.array([[60, 70, 65, 80], [20, 5, 10, 0], [10, 8, 7, 6]])
# Data for Mathematics and Japanese
b0 = np.array([[60, 70, 65, 80], [20, 5, 10, 0]])
# Data for English
b1 = np.array([10, 8, 7, 6])

# Give data at once.
cov2 = np.cov(b)
cov2 = cov2 * 3.0 / 4.0  # invariance variance to sample variance
print(cov2)
print('----------')
# Give data divided into two parts
cov21 = np.cov(b0, y=b1)
cov21 = cov21 * 3.0 / 4.0  # invariance variance to sample variance
print(cov21)

"""
Answer:
[[ 54.6875 -51.5625  -9.0625]
 [-51.5625  54.6875   9.6875]
 [ -9.0625   9.6875   2.1875]]
"""

"""
If the data is given in the form that the first row of the matrix 
represents the data for one person, use 'rowvar=False' option.
"""
a = np.array([[60, 20, 10], [70, 5, 8], [65, 10, 7], [80, 0, 6]])

print(a)
cov1 = np.cov(a, rowvar=False)
cov1 = cov1 * 3.0 / 4.0  # invariance variance to sample variance
print(cov1)
print('----------')
"""
Answer:
[[ 54.6875 -51.5625  -9.0625]
 [-51.5625  54.6875   9.6875]
 [ -9.0625   9.6875   2.1875]]
"""

"""
Set mean values and a covariance matrix, 
and generate sample points. 
"""
mean = np.array([0.1, 0.2, 0.3])
cov = np.array([
    [1.0, 0.2, 0.5],
    [0.2, 1.0, 0.7],
    [0.5, 0.7, 1.0]])

x, y, z = np.random.multivariate_normal(mean, cov, 100).T

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot(x, y, 'x')
plt.title("covariance=0.2")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
ax1 = fig.add_subplot(222)
ax1.plot(y, z, 'x')
plt.title("covariance=0.7")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
ax1 = fig.add_subplot(223)
ax1.plot(z, x, 'x')
plt.title("covariance=0.5")
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.show()

"""
Generate sample points.
and stack them vertically to create a data matrix.
"""
# Initialize
data_matrix = np.random.multivariate_normal(mean, cov, 1)

for i in range(9999):
    # Generate one sample.
    x = np.random.multivariate_normal(mean, cov, 1)
    data_matrix = np.vstack((data_matrix, x))


"""
Calculate a covariance matrix from the generated sample
points, and its eigen values and eigen vectors.
"""
cov = np.cov(data_matrix, rowvar=False)

# Get eigen values w[] and eigen vectors v[][]
w, v = np.linalg.eig(cov)
print('w = {}'.format(w))
print('v = {}'.format(v))
print('vT = {}'.format(v.T))

# eigen vectors corresponding to w[0], w[1] and w[2]
print('v0 = {}'.format(v[:,0]))
print('v1 = {}'.format(v[:,1]))
print('v2 = {}'.format(v[:,2]))

"""
Generate sample points and convert with the transformation matrix 
obtained above (KL-transformation).
"""
# Initialize
x = np.random.multivariate_normal(mean, cov, 1)
x1 = np.dot(x, v)

for i in range(999):
    x = np.random.multivariate_normal(mean, cov, 1)
    x1 = np.vstack((x1, np.dot(x, v)))

# Check the values!
for i in range(3):
    mean = np.mean(x1[:,i])
    var = np.var(x1[:,i])
    print('mean[{}]={}'.format(i, mean))  # mean should be zero.
    print('var[{}]={}'.format(i, var))  # var[i] should be eigen value[i]

# print('x1:\n', x1)
# print(x1[:,0])
# print(x1[:,1])

plt.plot(x1[:,0], x1[:,1], 'x')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

plt.plot(x1[:,1], x1[:,2], 'x')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

"""
w = [ 1.99779707  0.81551994  0.22727927]
v = [[-0.45117978 -0.82582666 -0.33830036]
 [-0.59050822  0.5604821  -0.58065468]
 [-0.66913141  0.06221051  0.74053562]]
vT = [[-0.45117978 -0.59050822 -0.66913141]
 [-0.82582666  0.5604821   0.06221051]
 [-0.33830036 -0.58065468  0.74053562]]
v0 = [-0.45117978 -0.59050822 -0.66913141]
v1 = [-0.82582666  0.5604821   0.06221051]
v2 = [-0.33830036 -0.58065468  0.74053562]
mean[0]=-0.3883181692231954
var[0]=1.929363121828424
mean[1]=0.05518271704779504
var[1]=0.82724402420096
mean[2]=0.0826800825144486
var[2]=0.24231061569748422
"""






