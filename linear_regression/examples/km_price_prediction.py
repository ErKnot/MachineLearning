from linear_regression.linear_regression import LinearRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

# loading the dataset
dataset_path = "linear_regression/examples/datasets/km_price.csv"
df = pd.read_csv(dataset_path)

# the trainig data
Xy = df.to_numpy().astype("float64")
print(Xy)
print("Dimensions of the dataset: ", Xy.shape)

# defining the treaning features and targets and changing the units 
X = Xy[:,:-1] / 100000 # km * e^5
y = Xy[:,-1:] / 1000 # price * e^3 

# fitting the LinearRegression class and computing the predictions on the training features
num_iteration = 300
lin_reg = LinearRegression(X, y)

# train the model and use it to predict the targets
lin_reg.fit(learning_rate = 0.1, n_iter=num_iteration)
print("Trained theta: ", lin_reg.theta)
y_trained_pred = lin_reg.predict(X)

# save the cost function history
cost_history = lin_reg.mse_training_history()


# fit the model analytically and use it to prdict the targets
theta = lin_reg.analytical_estimation(X, y)
print("theta: ", theta)
y_pred= theta[0] + theta[1] * X


# Plot the dataset with the linear regression models and the cost function history
fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

# plotting the data and the graph of the fitted linear regression
axs[0].scatter(X[:,-1:], y, label="training data")
axs[0].plot(X, y_trained_pred, color="red", label="trained linear regression model")
axs[0].plot(X, y_pred, color="green", label="linear regression model")
axs[0].legend()
axs[0].set_ylabel("price * e^3")
axs[0].set_xlabel("Km * e^5")
axs[0].set_title("Linear regression")

# plotting the graph of the mean square error
axs[1].plot([i for i in range(num_iteration)], cost_history,color="red", label="means square error")
axs[1].set_xlabel("num_iterations")
axs[1].set_ylabel("R")
axs[1].legend()
axs[1].set_title("Cost function history")
plt.show()
