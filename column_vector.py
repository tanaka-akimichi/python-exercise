import numpy as np

# row vector
b = np.array([1, 2])
print(b.ndim)
print(b.shape)

# Attribute T returns the same value if ndim = 1
print(b)
print(b.T)

# column vector
c = b[:, np.newaxis]
print(c)
print(c.ndim)
print(c.shape)

# Matrix A
a = np.array([[1,2], [3,4]])
a1 = a.T
print(a)
print(a1)

# y is equal to y1
y = np.dot(b, a)
print(y)
y1 = np.dot(a1, c)
print(y1)





