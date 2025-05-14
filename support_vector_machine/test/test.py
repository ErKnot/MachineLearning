import numpy as np


a = np.array([i for i in range(0, 5)])
b = np.array([i for i in range(0, 5)])
c = np.array([[j for j in range(0,5)] for i in range(0,5)])
print(a*b)
print(c)
d = a*b @ c
print(d.shape)
