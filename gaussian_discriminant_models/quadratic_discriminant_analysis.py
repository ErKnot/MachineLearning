import numpy as np

class GaussianQuadraticDiscriminant:
    """
    A Gaussian Quadratic Discriminant Analysis classifier, a model that fits to a training dataset and makes predictions. 
    The model assumes that each class follows a Gaussian (normal) distribution with its own covariance matrix. 

    Some mathematical simplifications are used to speed up computations. For example, the actual multivariate Gaussian distribution is never computed directly; instead, only the arguments of its exponentials are used. Additionally, when predicting the class label for a vector, we only consider the addends that vary across classes, ignoring constants that remain the same for all classes.
    """

    def __init__(self):
        """
        Initialize the class attributes.

        Attributes (just some of them for clarity):
            self.cov_matrices: three dimensional np.ndarray with the covariance matrices of all the classes. Each row corresponds to a class.
            self.means_matrix: the means of each feature for each class. Each row corresponds to a class.
            self.class_priors: The prior probabilities of each class (calculated as the number of elements in a class divided by the total number of samples).
        """
        self.n_sample = None
        self.n_features = None
        self.n_classes = None
        self.classes = None
        self.means_matrix = None
        self.class_priors = None
        self.cov_matrices= None
        self.predictions = None
        self._dtype = "float64"
        self._fit_method_called = False 

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Gaussian Quadratic Discriminant Analysis classifier to the training data.

        Args:
            X: the feature matrix of the training dataset.
            y: the target (class labels) of the training dataset.

        The model calculates:
            the means for each feature for each class.
            The covariance matrix of each class 
            The prior probability of each class.
        """

        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.means_matrix = np.zeros(shape=(self.n_classes, self.n_features), dtype=self._dtype)
        self.class_priors = np.zeros(shape=(self.n_classes), dtype=self._dtype)
        self.cov_matrices = np.zeros(shape=(self.n_classes, self.n_features, self.n_features), dtype=self._dtype)

        for num, cl in enumerate(self.classes):
            X_cl = X[y == cl]
            self.class_priors[num] = X_cl.shape[0] / self.n_samples
            self.means_matrix[num, :] = X_cl.mean(axis=0)

            dif = X_cl - self.means_matrix[num, :]
            matrix = np.zeros(shape=(self.n_features, self.n_features), dtype=self._dtype)

            for row in range(dif.shape[0]):
                matrix += np.outer(dif[row,:], dif[row,:])

            self.cov_matrices[num, :, :] = 1/ X_cl.shape[0] * matrix 

        # indicates that the fit method as been called successfully
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
        det_cov_matrices = np.ndarray(shape = (self.n_classes), dtype = self._dtype)
        inv_cov_matrices = np.ndarray(shape = (self.n_classes, self.n_features, self.n_features), dtype=self._dtype)

        for num, cl in enumerate(self.classes):
            det_cov_matrices[num] = np.linalg.det(self.cov_matrices[num,:,:])
            inv_cov_matrices[num, : :] = np.linalg.inv(self.cov_matrices[num,:,:])
            diff = X - self.means_matrix[num,:]
            
            arg_exp = np.sum((diff @ inv_cov_matrices[num, :,:]) * diff, axis = 1)
            ranks[:,num]= (-0.5 * np.log(det_cov_matrices[num]) - 0.5 * arg_exp) + np.log(self.class_priors[num])

        
        self.predictions = np.argmax(ranks, axis=1, keepdims=True)
        return self.predictions

