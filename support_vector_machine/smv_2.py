import numpy as np
from typing import Callable
def linear_kernel(v: np.ndarray,w: np.ndarray) -> float:
    """
    Linear kernel implementation.

    Arguments:
        v: a (m,) vector or a family of n vector of dimension m as a (n,m) matrix
        w: a (m,) vector or a family of n vector of dimension m as a (n,m) matrix
    
    Returns:
        the linear kernel:
            - It computes the dot product of v and w.T.

    """
    return v @ w.T




def compute_scores(x, X, targets, alphas, kernel, b):
    yalphas = targets * alphas
    return yalphas.reshape(1, -1) @ kernel(X, x) + b

def predict(x, X, targets, alphas, kernel, b):
    scores = compute_scores(x, X, targets, alphas, kernel, b)
    pred = np.where(scores < 0, -1, 1)
    return pred


def compute_errors(X, targets, alphas, kernel, b):
    scores = compute_score(X, X, targets, alphas, kernel, b)
    errors = scores - targets
    return errors


def compute_boundaries(a_1: float, a_2: float, y_1: int, y_2: int, c: float):
    """
    Compute the boundaries of the lagrangian multiplier a_2.

    Arguments:
        a_1: float, a lagrangian multiplier
        a_2: float, a lagrangian multiplier
        y_1: int, the target associated to a_1
        y_2: int, the target associated to a_2
        c: float, the control constant of the objective

    Returns:
        L: float, the lower boundary for a_2
        H: float, the highest boundary for a_2
        
    """
    if y_1 != y_2:
        L = np.max([0, a_2 - a_1])
        H = np.min([c, c + a_2 - a_1])

    else:
        L = np.max([0, a_2 + a_1 - c])
        H = np.min([c,a_2 + a_1])

    if L == H:
        return False 

    return L, H

def compute_eta(kernel: Callable, x_1: np.ndarray, x_2: np.ndarray) -> float:
    """
    Copute the second derivative of the objective function of the dual optimization problem defined for the support vector classifier.

    Arguments:
        kernel: Collable, a kernel function with two arguments.
        x_1: a (m,) vector
        x_2: a (m,) vector

    Returns:
        foat, the result of the second derivative
    """
    return kernel(x_1, x_1) + kernel(x_2, x_2) - 2 * kernel(x_1, x_2)


def compute_a_1_2(a_1: float, a_2: float,s: int, L: float, H: float, eta: float, err_1, err_2):

    """
    Compute the Lagrange multipliers a_1_cl and a_2_cl

    Arguments:
        a_1: float, the current Lagrange multiplier
        a_2: float, the current Lagrange multiplier
        s: int, the product of the targets associated to the Lagrange multipliers
        L: float, the lower boundary of a_2
        H: float, the higher boundary of a_2
        eta: float, the second derivative of the objective function
        err_1: float, the error of the training example associated to y_1
        err_2: float, the error of the training example associated to y_2

    Returns:
        The new optimized Lagrange multipliers a_1_cl and a_2_cl.
    """
    a_2_new = a_2 + (y_2 * (err_1 - err_2)) / eta
    a_2_cl = H if a_2_new >= H else a_2_new if L < a_2_new < H else L
    a_1_cl = a_1 + s * (a_2 - a_2_cl)

    return a_1_cl, a_2_cl

def check_kkt(X, targets, alphas, b, c, kernel):
    indexs = np.zeros(shape = (alphas.shape[0]))
    scores = compute_scores(X, X, targets, alphas, kernel, b)
    yscores = targets * scores

    cond1 = (np.isclose(alphas, 0, atol=0.01) & yscores < 1) 
    cond2 = (0 < alphas & alphas < c & np.isclose(yscores, 1, atol=0.01))
    cond3 = (np.isclose(alphas, c, atol=0.01) & yscores > 1)
    
    return np.where(cond1 | cond2 | cond3)[0]

def find_non_boundary_values(alphas, c, tol: float = 0.01):
    bound_0 = (alphas > tol)
    bound_c = (c - alphas > tol)
    return np.where(bound_0 | bound_c)

def choose_y1(y_2, X, tergets):


def fit():
    y2_list = check_kkt()

    if len(y2_list) == 0:
        return 0
    
    y2_non_boundary = find_non_boundary_values(alphas, c)
    if len(y2_non_boundaries) > 0:
        errors = compute_errors()
        for y2 in y2_non_boundaries:
            y2_err = compute_errors()
            abs_err = np.absolute(y2_err - errors)
            y_1 = np.argmax(abs_err)
            a_1, a_2 = compute_a_1_2
                return 1



        

    
                                                                                        

    


if __name__ == "__main__":

    alphas = np.array([1,1])
    targets = np.array([1,-1])
    X = np.array([[0,1],[1, 1]])
    predictions = predict(X,X, targets, alphas, linear_kernel, 0)
    print(predictions)
    

