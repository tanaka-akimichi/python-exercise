import numpy as np


x = 1
while x > 0:
    print('{} {}'.format(x, np.log(x)))
    x /= 2.0


