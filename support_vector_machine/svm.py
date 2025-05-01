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

class SVM:
    def __init__(self):
        self.vect = None
        self.intercept = None

    def fit(self, X:np.ndarray, vect: np.ndarray, intercept: float, lagrange_mlp: np.ndarray):
        pass

    def compute_score(self, x, X, targets, alphas, kernel, b):
        yalphas = targets * alphas
        return yalphas.reshape(1, -1) @ kernel(X, x) + b

    def compute_errors(self, X, targets, alphas, kernel, b):
        scores = compute_score


    

    def kkt_conditions(self, a, y, u, x, c):
        pass
        

    def compute_boundaries(self, a_1: float, a_2: float, y_1: int, y_2: int, c: float):
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
    
    def compute_eta(self, kernel: Callable, x_1: np.ndarray, x_2: np.ndarray) -> float:
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


    def compute_a_2(a_1: float, a_2: float,s: int, L: float, H: float, eta: float, err_1, err_2):

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


        
    def predict(self, X:np.ndarray) -> None:
        return X @ self.vect + self.intercept


