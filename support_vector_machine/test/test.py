import numpy as np

v = np.array(np.arange(0,4)).reshape(2,2)
w = np.array(np.arange(0,4)).reshape(2,2)
# print(v @ w)
# print(np.max([0,1]))

def fun():
    return 1, 2

a = fun()
print(a) 
a, b = fun()
print(a, b) 

    
