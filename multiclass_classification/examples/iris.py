import numpy as np
import pandas as pd

from multiclass_classification.multiclass_classification_model import MulticlassClassification
from multiclass_classification.utilities import onehotencoder, softmax

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

# Preprocessing

## loading the dataset
iris_path = "multiclass_classification/examples/datasets/Iris.csv"
iris_df = pd.read_csv(iris_path)

## drop useless coulms and replace the categories names with values
iris_df.drop(["Id"], axis=1, inplace=True)
species_map = { species: idx for idx, species in enumerate(iris_df["Species"].unique())}
iris_df_num = iris_df.replace(species_map)
iris_df_num = iris_df_num.astype("float64")
print(iris_df_num.corr())

## convert the dataset to a numpy array
Xy = iris_df_num.to_numpy()
X = Xy[:,:-1]
Y = onehotencoder(Xy[:,-1])

# Fitting the model and testing it on the same dataset
mc = MulticlassClassification()
mc.fit(X,Y, n_iter=10000)
predictions = mc.predict(X) 
# print(Xy[:,-1] == prediction)

print("List showing whether each input sample was classified correctly by comparing the predicted labels to the true labels:\n",[True if all(Y[row, :] == predictions[row,:]) else False for row in range(Y.shape[0])])

cost_training_history = mc.training_history()


fig, axs = plt.subplots()
axs.plot(range(len(cost_training_history)), cost_training_history, label="loss fucntion" )
axs.set_xlabel("num_iterations")
axs.set_ylabel("R")
axs.set_title("Cost function history")
plt.show()

