import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

from scipy.stats import multivariate_normal

from gaussian_discriminant_models.gaussian_naive_bayes import GaussianNaiveBayes

# Randomly generate a dataset fallowing multivariate gaussians random density functions.
# We define a mean vector and a covariance matrix for each class of datapoints.
# Since whe are testing Gaussian naive Bayes, in this example we will use diagonal covarian matrices. 
mean_1 = (-1, 2)
cov_1 = [[2,0], [0,1]]
class_1 = np.random.multivariate_normal(mean_1, cov_1, size=500)
y_1 = np.zeros(shape=(class_1.shape[0], 1))

mean_2 = (4, 0)
cov_2 = [[1,0], [0,3.5]]
class_2 = np.random.multivariate_normal(mean_2, cov_2, size=500)
y_2 = np.ones(shape=(class_2.shape[0], 1))

mean_3 = (-1, -4)
cov_3 = [[4,0], [0,2.75]]
class_3 = np.random.multivariate_normal(mean_3, cov_3, size=500)
y_3 = np.full(shape=(class_3.shape[0], 1), fill_value = 2)

# Concatenating the data points with thirs respective class vectors and then stacking them
# to create the dataset
Xy = np.r_["0", np.r_["-1", class_1, y_1], np.r_["-1", class_2, y_2], np.r_["-1", class_3, y_3]]

# Dividing the dataset in features and targets
X = Xy[:,:-1]
y = Xy[:,-1]
print(y)

gnb = GaussianNaiveBayes()
gnb.fit(X,y)
print("predicted means matrix:\n", gnb.means_matrix)
print("predicted variances_matrix:\n", gnb.variances_matrix)
print("predicted class priors:\n", gnb.class_priors)
# test_data = np.array([[1, 2], [3, 4], [6, 0], [-1, 3], [4, 0]])
# test_data = class_1
results = gnb.predict(X)
right_predictions = y == results.flatten()

print("Accuracy on the training set: ", np.sum(right_predictions) / right_predictions.shape[0])


# Plotting the data
# creating a grid of points for the contour lines
x, y = np.mgrid[-8:8:.1, -10:5:.1]
data = np.dstack((x,y))

# Computing the predicted gaussian density fucntions on the grid 
z = np.zeros(shape = (3, data.shape[0], data.shape[1]))
for cl in range(3):
    predicted_mult_norm = multivariate_normal(gnb.means_matrix[cl], np.diag(gnb.variances_matrix[cl]) )
    z[cl,:,:] = predicted_mult_norm.pdf(data)     

fig, ax = plt.subplots(1,2, figsize=(15, 7), layout="constrained")
# plotting the dataset
ax[0].set_title("Data set")
ax[0].set_ylabel("y")
ax[0].set_xlabel("x")
ax[0].plot(class_1[:,0], class_1[:,1], '.', c='r')
ax[0].plot(class_2[:,0], class_2[:,1], '.', c='b')
ax[0].plot(class_3[:,0], class_3[:,1], '.', c='m')
ax[0].grid()

# plotting the dataset with the countour lines of the predicted density functions
ax[1].set_title("Countour lines of the predicted prob. density functions")
ax[1].set_ylabel("y")
ax[1].set_xlabel("x")
ax[1].plot(class_1[:,0], class_1[:,1], '.', c='r')
ax[1].plot(class_2[:,0], class_2[:,1], '.', c='b')
ax[1].plot(class_3[:,0], class_3[:,1], '.', c='m')
ax[1].contour(x, y, z[0,:,:] )
ax[1].contour(x, y, z[1,:,:] )
ax[1].contour(x, y, z[2,:,:] )
ax[1].grid()
plt.show()

