import numpy as np

def onehotencoder(y:np.ndarray, n_classes: int = None):
    
    if n_classes == None:
        n_classes = len(np.unique(y))

    y_hotencoded = np.array([[1 if cl == y[row] else 0 for cl in range(n_classes)] for row in range(len(y))])
    return y_hotencoded



y = [1, 2, 0, 0]

result = onehotencoder(y, 3)
print(result)

