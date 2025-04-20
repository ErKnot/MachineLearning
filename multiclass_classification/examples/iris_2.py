import numpy as np
import pandas as pd

from multiclass_classification.multiclass_2 import MulticlassClassification
from multiclass_classification.utilities import onehotencoder, softmax

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
X = Xy[:,:-1]
Y = onehotencoder(Xy[:,-1])
print("Shape of X: ", X.shape)
theta = np.ones(shape=(4, 3))
mc = MulticlassClassification()
theta = mc.fit(X,Y, n_iter=10000)
prediction = mc.predict(X) 

print(Xy[:,-1] == prediction)

