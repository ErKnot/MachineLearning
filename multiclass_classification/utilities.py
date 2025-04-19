import numpy as np

def onehotencoder(y:np.ndarray|list, classes_map: dict = None) -> np.ndarray:
    """
    Create a one hot encoded nparray
    """ 
    _y = np.array(y).ravel()

    if classes_map == None:
        classes = np.unique(_y)
        classes_map = {key:val for  val, key in enumerate(classes)}
    
    y_hotencoded = np.array([[1 if cl == classes_map[_y[row]] else 0 for cl in range(len(classes_map))] for row in range(len(_y))])
    return y_hotencoded


def softmax(X: np.ndarray, axis: int = 0) -> np.ndarray:
    exp = np.exp(X)
    sum_exp = np.sum(exp, axis=axis, keepdims=True)
    return exp / sum_exp


