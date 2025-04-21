import numpy as np

class GaussianLinearDiscriminant:
    """
    A Gaussian linear discriminant model that fits to a training dataset and makes predictions. 
    The model assumes that each class follows a Gaussian (normal) distribution for each feature with a common covariant matrix for all classes.

    Some mathematical simplifications are used to speed up computations. For example, the actual multivariate Gaussian distribution is never computed directly; instead, only the arguments of its exponentials are used. Additionally, when predicting the class label for a vector, we only consider the addends that vary across classes, ignoring constants that remain the same for all classes.
    """

    def __init__(self):
        self.n_sample = None
        self.n_features = None
        self.n_classes = None
        self.classes = None
        self.means_matrix = None
        self.class_priors = None
        self.cov_matrix= None
        self.predictions = None
        self._dtype = "float64"
        self._fit_method_called = False 
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Gaussian linear discriminant model to the training data.

        Args:
            X: the feature matrix of the training dataset.
            y: the target (class labels) of the training dataset.

        The model calculates:
            the means for each feature for each class.
            The covariance matrix which is the same for all the classes.
            The prior probability of eaxh class.
        """
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.means_matrix = np.zeros(shape=(self.n_classes, self.n_features), dtype=self._dtype)
        self.class_priors = np.zeros(shape=(self.n_classes), dtype=self._dtype)
        self.cov_matrix = np.zeros(shape= (self.n_features, self.n_features), dtype=self._dtype)
        self.cov_matrix_addends = np.zeros(shape= (self.n_classes, self.n_features, self.n_features), dtype=self._dtype)

        for num, cl in enumerate(self.classes):
            X_cl = X[y == cl]
            self.class_priors[num] = X_cl.shape[0] / self.n_samples
            self.means_matrix[num, :] = X_cl.mean(axis=0)
            
            diff = X_cl - self.means_matrix[num, :]
            matrix = np.zeros(shape=(self.n_features, self.n_features), dtype=self._dtype)

            for row in range(diff.shape[0]):
                matrix += np.outer(diff[row,:], diff[row,:])

            self.cov_matrix_addends[num,:,:] = matrix 

        self.cov_matrix = 1 / self.n_samples * self.cov_matrix_addends.sum(axis = 0)

        # indicate that the fit method as been called successfully
        self._fit_method_called = True


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for a given set of feature vectors.

        Args:
            X: a set of feature vectors to classify.

        Returns:
            np.ndarray: an array of predicted class labels for each vector in X.
        """

        # Check if the model has been fit
        if not self._fit_method_called:
            raise RuntimeError("The method predict can not be called without calling the fit method sucessfully first.")

        ranks = np.zeros(shape=(X.shape[0], self.n_classes), dtype=self._dtype)
        det_cov_matrix = np.linalg.det(self.cov_matrix)
        inv_cov_matrix = np.linalg.inv(self.cov_matrix)

        for num, cl in enumerate(self.classes):
            diff = X - self.means_matrix[num,:]
            # we compute the arguments of the exponential of the gaussian density fucntion
            # for each feature vector
            arg_exp = np.sum((diff @ inv_cov_matrix) * diff, axis = 1)
            ranks[:,num] = (-0.5*np.log(det_cov_matrix)- 0.5 * arg_exp) + np.log(self.class_priors[num])
        self.predictions = np.argmax(ranks, axis=1, keepdims=True)
        return self.predictions
            


