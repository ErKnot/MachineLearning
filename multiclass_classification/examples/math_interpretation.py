
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

# For this example we keep two features that seems sufficients to specify the classes
df = iris_df_num.drop(["SepalLengthCm", "SepalWidthCm"], axis=1)
print(df.info())


Xy = df.to_numpy()
X = Xy[:,:-1]
Y = onehotencoder(Xy[:,-1])

mc = MulticlassClassification()
mc.fit(X,Y, n_iter=10000)

predictions = mc.predict(X) 
# print(Xy[:,-1] == prediction)
correct_pred = [True if all(Y[row, :] == predictions[row,:]) else False for row in range(Y.shape[0])]
print("List showing whether each input sample was classified correctly by comparing the predicted labels to the true labels:\n",correct_pred)
print("% correct predictions:\n", np.sum(correct_pred)/ len(correct_pred))
pred_cl = np.argmax(predictions, axis=1, keepdims=True).ravel()
print(pred_cl)

data_pr_0 = Xy[:,:-1][pred_cl==0]
data_pr_1 = Xy[:,:-1][pred_cl==1]
data_pr_2 = Xy[:,:-1][pred_cl==2]
theta = mc.theta
m_01 = -(theta[0,0] - theta[0,1]) / (theta[1,0] - theta[1,1])
m_02 = -(theta[0,0] - theta[0,2]) / (theta[1,0] - theta[1,2])
m_12 = -(theta[0,1] - theta[0,2]) / (theta[1,1] - theta[1,2])
print(m_01)
t = np.arange(0, 9, 0.1)

data_0 = Xy[:,:-1][Xy[:,-1]==0]
y_0 = Xy[:,-1][Xy[:,-1]==0]
data_1 = Xy[:,:-1][Xy[:,-1]==1]
y_1 = Xy[:,-1][Xy[:,-1]==1]
data_2 = Xy[:,:-1][Xy[:,-1]==2]
y_2 = Xy[:,-1][Xy[:,-1]==2]

fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].set_title("Training data + predicted classification boundaries")
ax[0].set_ylabel("Petal Lenght (Cm)")
ax[0].set_xlabel("Petal Width (Cm)")
ax[0].plot(data_0[:,0],data_0[:,1], '.', c='r')
ax[0].plot(data_1[:,0],data_1[:,1], '.', c='b')
ax[0].plot(data_2[:,0],data_2[:,1], '.', c='m')
ax[0].plot(t, m_01*t)
ax[0].plot(t, m_02*t)
ax[0].plot(t, m_12*t)


ax[1].set_title("Predicted data + predicted classification boundaries")
ax[1].set_ylabel("Petal Lenght (Cm)")
ax[1].set_xlabel("Petal Width (Cm)")
ax[1].plot(data_pr_0[:,0],data_pr_0[:,1], '.', c='r')
ax[1].plot(data_pr_1[:,0],data_pr_1[:,1], '.', c='b')
ax[1].plot(data_pr_2[:,0],data_pr_2[:,1], '.', c='m')
ax[1].plot(t, m_01*t)
ax[1].plot(t, m_02*t)
ax[1].plot(t, m_12*t)
plt.show()


