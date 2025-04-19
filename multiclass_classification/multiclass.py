import numpy as np
from multiclass_classification.gradient_descent import stochastic_gradient_descent

def softmax(x : np.ndarray, axis: int = 0):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    # print(sum_exp_x)
    return exp_x / sum_exp_x

class MulticlassClassification:
    """
    Multi-class classification model.
In a classification problem with C classes, each feature vector is multiplied (via dot product) by C weight vectors. The softmax function is then applied to the resulting values. The C output values represent the probabilities that the feature vector belongs to each of the C classes.
    """ 
    def __init__(self, X: np.ndarray, y:np.ndarray, num_classes: int, dtype: str = "float64"):
        # check X, y dimensions
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"The variables 'X' and 'y' must have the same number of rows. While 'X' has {X.shape[0]} rows, and 'y' {y.shape[0]}.")

        # instantiate the attributes
        self._dtype = dtype
        self._X = X.astype(dtype=self._dtype) 
        self.y = y.astype(dtype=self._dtype) 
        self.num_classes = num_classes
        self._fit_method_called = None

    def fit(self, learning_rate: float = 0.01, batch_size: int = None, n_iter: int = 100, random_seed: int = None) -> None:


        """
        Uses a gradient descent algorithm to fit the parameters of a multi-class classification model from a training set.
        If you want to use a stochastic gradient descent define a strictly positive 'batch_size'.
        """
        self.theta = np.ones(shape=(self._X.shape[1], self.num_classes), dtype=self._dtype)

        self.theta, self._theta_training_history = stochastic_gradient_descent(
                        vector=self.theta,
                        learning_rate=learning_rate,
                        gradient=MulticlassClassification.grad_loss_function,
                        X=self._X, 
                        y=self.y,
                        batch_size=batch_size,
                        n_iter=n_iter,
                        random_seed=random_seed
                        )

        # set the attribute that check if the method has been successfully called to True
        self._fit_method_called = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the image of the fitted multiclass-classification model.
        """ 
        # check if the model has been fitted
        if not self._fit_method_called:
            raise RuntimeError("The method predict can not be called without calling the fit method sucessfully first.")

        Xtheta = -X @ self.theta
        prob = softmax(Xtheta, axis=1)
        max = np.max(prob, axis=1, keepdims=True)
        predictions = np.where(prob == max, 1, 0)
        return predictions

    def loss_training_history(self) -> list:

        # check if the model has been fitted
        if not self._fit_method_called:
            raise RuntimeError("The method predict can not be called without calling the fit method sucessfully first.")

        loss_training_history = [MulticlassClassification.loss_function(theta, self._X, self.y) for theta in self._theta_training_history]
        return loss_training_history

    @staticmethod
    def loss_function(theta, X, Y):
        Xtheta = -X @ theta
        num_trainings = X.shape[0]
        result = 1 / num_trainings * (np.trace(X @ theta @ Y.T) + np.sum(np.log(np.sum(np.exp(Xtheta), axis = 1))))
        return result

    @staticmethod
    def grad_loss_function(theta, X, Y):
        Xtheta = -X @ theta
        Prob = softmax(Xtheta, axis=1)
        num_trainings = X.shape[0]
        return 1/num_trainings * (X.T @ (Y - Prob))
