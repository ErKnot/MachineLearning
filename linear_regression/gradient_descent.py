import numpy as np
from collections.abc import Callable

def stochastic_gradient_descent(vector: np.ndarray, learning_rate: float, gradient: Callable, X: np.ndarray, y: np.ndarray, batch_size: int = None, n_iter: int = 100, random_seed: int = None) -> np.ndarray:
    """
    Compute the gradient descent algorithm.
    To use the stochastic gradient descent set the 'batch_size' variable as an integer positive number smaller than the number of rows of t 

    Args:
        vector: the vector used to start the algorithm
        learning_rate: learning rate for the (stochastic) gradient descent
        gradient: the gradient of the function to which apply the algorithm
        X: features of the training set
        y: targets of the training set
        batch_size: batch size for the stochastic gradint discent
        n_iter: number of iterations of the (stochastic) gradient descent
        random_seed: seed for reproducing the experiment. It is used when shaffling the training set.

    Returns:
        The vector obtain from the (stochastic) gradient descent algorithm
    """    
    
    # check if 'learning_rate' is a float (we will allow negative values and with any absolute value)
    if not isinstance(learning_rate, float):
        raise ValueError("The learning_rate variable must be a float number.")

    # check if 'n_iter' is a strictly positive integer
    if not isinstance(n_iter, int) or n_iter <=0:
        raise ValueError("The number of iterations 'n_iter' must be a strictly positive integer")

    # initiating the grad_disc_history list
    grad_disc_history = []

    # Classic gradient descent if batch_size is not define or bigger than the cardinality of the training set
    if not batch_size or batch_size >= X.shape[1]:
        for _ in range(n_iter):
            vector += -learning_rate * gradient(vector, X, y)
            grad_disc_history.append(vector.copy())

        return vector, grad_disc_history
   
    # if defined, set the random number generator using the random seed
    seed = random_seed if random_seed else None
    numpy_rng = np.random.default_rng(seed=seed)

    # initializing the training set
    Xy = np.r_['-1', X, y]

    for _ in range(n_iter):
        #shuffling the training set before going through the batches
        np.random.shuffle(Xy)
        
        for i in np.arange(0, Xy.shape[1], batch_size):
            vector -= learning_rate * gradient(vector, Xy[i:batch_size + i,:-1], Xy[i:batch_size + i, -1:])
            grad_disc_history.append(vector.copy())

    return vector, grad_disc_history
