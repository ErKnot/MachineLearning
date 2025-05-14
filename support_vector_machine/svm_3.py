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

def compute_boundaries(a1: float, a2: float, y1: int, y2: int, c: float, tol = 1e-3):
    """
    Compute the boundaries of the lagrangian multiplier a2.

    Arguments:
        a1: float, a lagrangian multiplier
        a2: float, a lagrangian multiplier
        y1: int, the target associated to a1
        y2: int, the target associated to a2
        c: float, the control constant of the objective

    Returns:
        L: float, the lower boundary for a2
        H: float, the highest boundary for a2
        
    """
    if y1 != y2:
        L = np.max([0, a2 - a1])
        H = np.min([c, c + a2 - a1])

    else:
        L = np.max([0, a2 + a1 - c])
        H = np.min([c,a2 + a1])

    if np.abs(L - H) < tol:
        return None, None 

    return L, H

def compute_eta(kernel: Callable, i1: np.ndarray, i2: np.ndarray) -> float:
    """
    Copute the second derivative of the objective function of the dual optimization problem defined for the support vector classifier.

    Arguments:
        kernel: Collable, a kernel function with two arguments.
        i1: a (m,) vector
        i2: a (m,) vector

    Returns:
        foat, the result of the second derivative
    """
    return kernel(i1, i1) + kernel(i2, i2) - 2 * kernel(i1, i2)

def compute_a1_2(a1: float, a2: float,s: int, L: float, H: float, eta: float, err_1, err_2):

    """
    Compute the Lagrange multipliers a1_cl and a2_cl

    Arguments:
        a1: float, the current Lagrange multiplier
        a2: float, the current Lagrange multiplier
        s: int, the product of the targets associated to the Lagrange multipliers
        L: float, the lower boundary of a2
        H: float, the higher boundary of a2
        eta: float, the second derivative of the objective function
        err_1: float, the error of the training example associated to y_1
        err_2: float, the error of the training example associated to y_2

    Returns:
        The new optimized Lagrange multipliers a1_cl and a2_cl.
    """
    a2_new = a2 + (y_2 * (err_1 - err_2)) / eta
    a2_cl = H if a2_new >= H else a2_new if L < a2_new < H else L
    a1_cl = a1 + s * (a2 - a2_cl)

    return a1_cl, a2_cl

def compute_threshold(e1, e2, x1, x2, y1, y2, a1, a1_new, a2, a2_new, kernel, c, tol, b):
    b1 = e1 + y1 * (a1_new - a1) * kernel(x1, x2) + y2 * (a2_new - a2) * kernel(x1, x2) + b
    b2 = e2 + y1 * (a1_new - a1) * kernel(x1, x2) + y2 * (a2_new - a2) * kernel(x1, x2) + b

    if tol < a1 < c - tol:
        return b1

    if tol < a2 < c - tol:
        return b2

    return (b1 + b2) / 2


def take_step(i1, i2, trainings, targets, alphas,E_chache, b, c = 1, tol = 1e-3):
    print("i1 is:", i1)
    print(i2)

    if i1 == i2:
        return 0
    
    x1 = trainings[i1]
    x2 = trainings[i2]
    y1 = target[i1]
    y2 = target[i2]
    a1 = alphas[i1]
    a2 = alphas[i2]
    E1 = err_cache[i1]
    E2 = err_cache[i2]
    
    s = y1 * y2
    L, H = compute_boundaries(a1, a2, y1, y2, c)    
    
    if L is None:
        return 0

    eta = compute_eta(kernel, x1, x2)
    if eta > 0:
        return 0

    a1_new, a2_new = compute_a1_2(a1, a2, s, L, H, eta, E1, E2)

    if np.abs(a2_new - a2) < tol:
        return 0

    alphas[i1] = a1_new
    alphas[i2] = a2_new
    b = compute_threshold(E1, E2, x1, x2, y1, y2, a1, a1_new, a2, a2_new, kernel, c, tol, b)
    err_cache = compute_score(training) - targets

    return 1

def find_nb(targets, alphas, kernel, c = 1, tol = 0.01):
    bound0 = (alphas > -tol)  
    boundC = (c - alphas > tol)
    return np.where(bound0 & boundC)[0]

def second_choice(E_nb, E2):
    absE = np.abs(E2 - E_nb)
    print("absE is:", absE)
    return np.argmax(absE)



def examin_example(i2, training, targets, alphas, err_cache, kernel, c = 1, tol = 1e-3):

    y2 = targets[i2]
    # print("y2", y2)
    a2 = alphas[i2]
    E2 = err_cache[0][i2] 
    # print("E2: ", E2)
    r2 = E2 * y2
    # print(r2)
    # print(type(r2))

    nb_idx = find_nb(targets, alphas, kernel, c, tol)
    # E_nb = predict(training[non_bounds,:]) - targets[non_bounds]

    if (r2 < -tol and a2 < c) or (r2 > tol and a2 > 0):
        # Choose the Lagrange multipliers that do not belongs to the boundry of [0, c]
        if len(nb_idx) > 1:
            i1 = second_choice(err_cache, E2)
            i1 = training[nb_idx[i1]]
            if take_step(i1, i2, training, targets, alphas,E2, c):
                return 1

        # If no ggood candidate for i1 is found in the previous loop, it loops over le non  boud Lagrange multipliers
        for i1 in np.random.permutation(nb_idx):
            if take_step(i1, i2, training, targets, alphas,E2, c):
                return 1
        # If there aren't non bound Lagrange multipliers it loops over all the multipliers
        for i1 in np.random.permutation(len(targets)):
            if take_step(i1, i2, targets, alphas,E2, c):
                return 1

    return 0
             
def main(training, targets, alphas, b, kernel, tol, c):

    num_changed = 0
    examin_all = 1
    err_cache = compute_scores(training, training, targets, alphas, kernel, b) - targets
    print(err_cache)

    while num_changed > 0 or examin_all:
        num_changed = 0
        # create the function the check which elements are not in the boundary
        non_bd = find_nb(targets, alphas, kernel)
        if examin_all:
            for i2 in range(len(targets)):
                num_changed += examin_example(i2, training, targets, alphas, err_cache, kernel)

        elif len(non_bd)>0:
            for index in non_boundary:
                num_changed += examin_example(i2, training, targets, alphas, err_cache, kernel)

        if examin_all == 1:
            examin_all = 0
        elif num_changed == 0:
            examin_all = 0

        if examin_all == 1:
            examin_all = 0
        elif num_changed == 0:
            examin_all = 1


if __name__ == "__main__":
    training = np.array([[2,1], [3,2.5], [2,4],[1,3]])
    targets = np.array([-1,-1,1,1])
    alphas = np.zeros(shape=(4))
    b = 0
    tol = 1e-3
    c = 1

    main(training, targets, alphas, b, linear_kernel, tol, c)

