import numpy as np
from grd import stochastic_gradient_descent

def SE_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.dot(X.T, (y - np.dot(X, theta)))

X = np.array([1,2,3,4]).reshape(2, -1)
y = np.array([5, 6]).reshape(-1, 1)
theta = np.array([1, 2], dtype=float).reshape(-1, 1)
print("X: ", X)
print("y: ", y)
print(SE_gradient(theta, X, y))

theta = stochastic_gradient_descent(theta = theta, learning_rate=0.1, gradient=SE_gradient, X=X, y=y)
print(theta)

# Xy = np.r_["-1", X, y]
# np.random.shuffle(Xy)
# print(Xy)
