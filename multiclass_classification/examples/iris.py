import numpy as np
import pandas as pd

from multiclass_classification.multiclass import MulticlassClassification

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

# Preprocessing

## Loading the dataset and converting it to a numpy array

## loading the dataset
iris_path = "multiclass_classification/examples/datasets/Iris.csv"
iris_df = pd.read_csv(iris_path)

## drop useless coulms and replace the categories names with values
iris_df.drop(["Id"], axis=1, inplace=True)
species_map = { species: idx for idx, species in enumerate(iris_df["Species"].unique())}
iris_df_num = iris_df.replace(species_map)
iris_df_num = iris_df_num.astype("float64")

# convert the dataset to a numpy array
Xy = iris_df_num.to_numpy()
e_1 = [1, 0, 0]
e_2 = [0, 1, 0]
e_3 = [0, 0, 1]

Y_hot_encoded= [e_1 if Xy[row, -1] == 0 else e_2 if Xy[row, -1] == 1 else e_3  for row in range(Xy.shape[0]) ]
dummies = np.array(Y_hot_encoded)
XY = np.r_['-1', Xy[:,:-1], Y_hot_encoded]

## Dfinining the features and targets matricies X and Y 
X = XY[:,:-3:]
Y = XY[:,-3:]

# Defining the training features and targets
X = Xy[:,:-1]
y = Xy[:,-1:]
print("shpae of x:", X.shape)
print("shape of y:", y.shape)



# Training the multiclass classification model
num_iterations = 1000 
mc = MulticlassClassification(X, Y, num_classes = 3)
mc.fit(X, y)
mc.fit(learning_rate = float(0.1), n_iter=num_iterations)
predictions = mc.predict(X)
## If you want to see the results in the form of the species map, uncomment the next line
# print(np.argmax(predictions, axis=1))

# Checking the predictions using the training set
print("List showing whether each input sample was classified correctly by comparing the predicted labels to the true labels:\n",[True if all(Y[row, :] == predictions[row,:]) else False for row in range(Y.shape[0])])


loss_training_history = mc.loss_training_history()

fig, axs = plt.subplots()
axs.plot(range(len(loss_training_history)), loss_training_history, label="loss fucntion" )
axs.set_xlabel("num_iterations")
axs.set_ylabel("R")
axs.set_title("Cost function history")
plt.show()
