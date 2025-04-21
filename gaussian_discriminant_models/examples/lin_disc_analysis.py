import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

from scipy.stats import multivariate_normal

from gaussian_discriminant_models.linear_discriminant_analysis import GaussianLinearDiscriminant 

# define the mean vector and the covariant matrix (semidefined positive)
cov = [[2,0.8], [0.8,1.5]]

mean_1 = (-2, 2)
class_1 = np.random.multivariate_normal(mean_1, cov, size=500)
y_1 = np.zeros(shape=(class_1.shape[0], 1))

mean_2 = (2, -0.5)
class_2 = np.random.multivariate_normal(mean_2, cov, size=500)
y_2 = np.ones(shape=(class_2.shape[0], 1))

mean_3 = (4, 4)
class_3 = np.random.multivariate_normal(mean_3, cov, size=500)
y_3 = np.full(shape=(class_3.shape[0], 1), fill_value=2)


# Concatenating the data points with thirs respective class vectors and then stacking them
# to create the dataset
Xy = np.r_["0", np.r_["-1", class_1, y_1], np.r_["-1", class_2, y_2], np.r_["-1", class_3, y_3]]
X = Xy[:,:-1]
y = Xy[:,-1]

gld = GaussianLinearDiscriminant()
gld.fit(X,y)
print("Fitted covariance matrix:\n",gld.cov_matrix)
print("Fitted mean vectors:\n",gld.means_matrix)
results = gld.predict(X)
print("Shape of the predictions: ",results.shape)
class_pred_1 = Xy[results.ravel() == 0, :]
class_pred_2 = Xy[results.ravel() == 1]
class_pred_3 = Xy[results.ravel() == 2]
# vector with true every time the prediction was correct
right_predictions = y == results.flatten()

print("Accuracy on the training set: ", np.sum(right_predictions) / right_predictions.shape[0])

# Plotting the data
# creating a grid of points for the contour lines
x, y = np.mgrid[-8:8:.1, -5:8:.1]
data = np.dstack((x,y))

# Computing the predicted gaussian density fucntions on the grid 
z = np.zeros(shape = (3, data.shape[0], data.shape[1]))
for cl in range(3):
    predicted_mult_norm = multivariate_normal(gld.means_matrix[cl], gld.cov_matrix)
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
ax[1].set_title("Countour lines of the predicted prob. density functions and\n predictions for the training set")
ax[1].set_ylabel("y")
ax[1].set_xlabel("x")
ax[1].plot(class_pred_1[:,0], class_pred_1[:,1], '.', c='r')
ax[1].plot(class_pred_2[:,0], class_pred_2[:,1], '.', c='b')
ax[1].plot(class_pred_3[:,0], class_pred_3[:,1], '.', c='m')
ax[1].contour(x, y, z[0,:,:] )
ax[1].contour(x, y, z[1,:,:] )
ax[1].contour(x, y, z[2,:,:] )
ax[1].grid()
plt.show()

#
