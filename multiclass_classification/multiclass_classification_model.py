import numpy as np
from linear_regression.gradient_descent import stochastic_gradient_descent
from multiclass_classification.utilities import softmax
from multiclass_classification.gradient_descent import stochastic_gradient_descent



class MulticlassClassification:
    """
    A multi-class classification model based on a linear approach.

    This model assumes that the target distribution is categorical (i.e., a multinomial distribution with one trial per example).
    Given a classification problem with C classes and a feature vector, the model computes the dot product between the input vector and C class-specific weight vectors. This results in a C-dimensional output vector, where each component represents an unnormalized score for a class. Applying the softmax function to this vector yields the class probabilities.

    The `fit` method learns the C weight vectors using the provided training data.

    For prediction, we make use of mathematical simplifications: since all softmax components are passed through the same
    strictly increasing function, the relative ranking of class scores is preserved. Thus, we can predict the most likely class
    without computing the actual probabilities.
    """
    def __init__(self):
        self._X = None
        self.y = None
        self.n_classes = None
        self.classes = None
        self.theta = None
        self._theta_training_history = None
        self._dtype = "float64"
        self._fit_method_called = False 

    def fit(self, X: np.ndarray, Y: np.ndarray, learning_rate: float = 0.1, batch_size: int = None, n_iter: int = 100, random_seed: int = None) -> None:
        """
        Fit the weight vectors of the classification model using (stochastic) gradient descent.

        This method optimizes the model parameters based on the training data. It also records
        the history of the weight vectors at each iteration in a list for later inspection or analysis.
        """        

        self._X = X
        self.y = Y
        self.n_classes = Y.shape[1]
        self.theta = np.ones(shape=(self._X.shape[1], self.n_classes), dtype=self._dtype)

        self.theta, self._theta_training_history = stochastic_gradient_descent(
                        vector=self.theta,
                        learning_rate=learning_rate,
                        gradient=self.gradient_cost_function,
                        X=self._X, 
                        y=self.y,
                        batch_size=batch_size,
                        n_iter=n_iter,
                        random_seed=random_seed
                        )
        
        self._fit_method_called = True

    def predict(self, X: np.array, keepdims: bool = False) -> np.array:
        """
        Given a matrix of feature vectors, compute the class scores for each input.

        This method returns the predicted class for each input vector by taking the argmax of the score vector (i.e., the index of the highest score), which corresponds to the most likely class.
        """
        # Check if the model has been fit
        if not self._fit_method_called:
            raise RuntimeError("The method predict can not be called without calling the fit method sucessfully first.")

        scores = -X @ self.theta
        scores_max = np.max(-X @ self.theta, axis=1, keepdims=True)
        return np.where(scores == scores_max, 1, 0) 

    def cost_function(self, theta:np.ndarray, X:np.ndarray, Y:np.array) -> float:
        Xtheta = - X @ theta
        sm_Xtheta = softmax(Xtheta, axis=1)
        return - np.trace(np.log(sm_Xtheta @ Y.T))

    def gradient_cost_function(self, theta:np.ndarray, X:np.ndarray, Y:np.array) -> np.ndarray:
        """
        The cost function used for fitting the model. To avoid overflow we use the mean of the gradient of the likelihood function.
        """
        Xtheta = - X @ theta
        sm_Xtheta = softmax(Xtheta, axis=1)
        return - 1/X.shape[0] * X.T @ (sm_Xtheta - Y)
    
    def training_history(self) -> list:
        # check if the model has been fitted
        if not self._fit_method_called:
            raise RuntimeError("The method predict can not be called without calling the fit method sucessfully first.")

        cost_training_history = [self.cost_function(theta, self._X, self.y) for theta in self._theta_training_history]
        return cost_training_history
