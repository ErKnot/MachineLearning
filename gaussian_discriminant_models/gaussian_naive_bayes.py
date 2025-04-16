import numpy as np

class GaussianNaiveBayes:
    """
    A Gaussian Naive Bayes model that fits to a training dataset and makes predictions. 
    The model assumes that each class follows a Gaussian (normal) distribution for each feature with a diagonal matrix as covariant matrix.

    Some mathematical simplifications are used to speed up computations. For example, the actual multivariate Gaussian distribution is never computed directly; instead, only the arguments of its exponentials are used. Additionally, when predicting the class label for a vector, we only consider the addends that vary across classes, ignoring constants that remain the same for all classes.
    """
    def __init__(self):
        """
        Initialize the class attributes.

        Attributes (just some of them for clarity):
            self.variances_matrix: each row corresponds to the diagonal of the covariance matrix of a class. Each row correspond to a class. 
            self.means_matrix: the means of each feature for each class. Each row corresponds to a class.
            self.class_priors: The prior probabilities of each class (calculated as the number of elements in a class divided by the total number of samples).
        """
        self.n_sample = None
        self.n_features = None
        self.n_classes = None
        self.classes = None
        self.means_matrix = None
        self.class_priors = None
        self.variances_matrix = None
        self.predictions = None
        self._dtype = "float64"
        self._fit_method_called = None


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Gaussian Naive Bayes model to the training data.

        Args:
            X: the feature matrix of the training dataset.
            y: the target (class labels) of the training dataset.

        The model calculates:
            the means for each feature for each class.
            The variance of each feature for each class.
            The prior probability of eaxh class.
        """
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.means_matrix = np.zeros(shape=(self.n_classes, self.n_features), dtype=self._dtype)
        self.class_priors = np.zeros(shape=(self.n_classes), dtype=self._dtype)
        self.variances_matrix = np.zeros(shape=(self.n_classes, self.n_features), dtype=self._dtype)

        for num, cl in enumerate(self.classes):
            X_cl = X[y == cl]
            self.class_priors[num] = X_cl.shape[0] / self.n_samples
            self.means_matrix[num, :] = X_cl.mean(axis=0)
            diff_square = (X_cl - self.means_matrix[num, :])**2
            self.variances_matrix[num,:] = diff_square.mean(axis=0)
            
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

       
        probs = np.zeros(shape=(X.shape[0], self.n_classes), dtype=self._dtype)
        det_var_matrix = np.zeros(shape=(self.n_classes), dtype=self._dtype)

        for num, cl in enumerate(self.classes):
            det_var_matrix[num] = np.prod(self.variances_matrix[num,:])
            sum_diff_square = np.sum((X - self.means_matrix[num, :])**2, axis=1)
            probs[:,num] = (-0.5*np.log(det_var_matrix[num]) - 0.5 * sum_diff_square) + np.log(self.class_priors[num])

        self.predictions = np.argmax(probs, axis=1, keepdims=True)
        return self.predictions


