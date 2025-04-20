
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

# Prediction without interceptor
mc = MulticlassClassification()
mc.fit(X,Y, n_iter=10000)
predictions = mc.predict(X) 
correct_pred = [True if all(Y[row, :] == predictions[row,:]) else False for row in range(Y.shape[0])]
# print("List showing whether each input sample was classified correctly by comparing the predicted labels to the true labels:\n",correct_pred)
print("% correct predictions without interceptor:\n", np.sum(correct_pred)/ len(correct_pred))
pred_cl = np.argmax(predictions, axis=1, keepdims=True).ravel()

# Predicted classes for the training set
data_pr_0 = Xy[:,:-1][pred_cl==0]
data_pr_1 = Xy[:,:-1][pred_cl==1]
data_pr_2 = Xy[:,:-1][pred_cl==2]
theta = mc.theta
m_01 = -(theta[0,0] - theta[0,1]) / (theta[1,0] - theta[1,1])
m_02 = -(theta[0,0] - theta[0,2]) / (theta[1,0] - theta[1,2])
m_12 = -(theta[0,1] - theta[0,2]) / (theta[1,1] - theta[1,2])
print(m_01)
t = np.arange(0, 9, 0.1)

# Predictions with interceptor
Xyi = np.insert(Xy, 0, 1, axis=1) 
Xi = Xyi[:,:-1]
Y = onehotencoder(Xy[:,-1])

mci = MulticlassClassification()
mci.fit(Xi,Y, n_iter=10000)
predictions_itc = mci.predict(Xi) 
correc_pred_itc = [True if all(Y[row, :] == predictions_itc[row,:]) else False for row in range(Y.shape[0])]
# print("List showing whether each input sample was classified correctly by comparing the predicted labels to the true labels:\n",correc_pred_itc)
print("% correct predictions with interceptor:\n", np.sum(correc_pred_itc)/ len(correc_pred_itc))
pred_cl_itc = np.argmax(predictions_itc, axis=1, keepdims=True).ravel()

data_itc_pr_0 = Xy[:,:-1][pred_cl_itc==0]
data_itc_pr_1 = Xy[:,:-1][pred_cl_itc==1]
data_itc_pr_2 = Xy[:,:-1][pred_cl_itc==2]
theta_itc = mci.theta

b_01 = (theta_itc[0,0] - theta_itc[0,1])/ (theta_itc[2,1]- theta_itc[2,0])
n_01 = (theta_itc[1,0] - theta_itc[1,1])/(theta_itc[2,1]- theta_itc[2,0])
b_02 = (theta_itc[0,0] - theta_itc[0,2])/ (theta_itc[2,2]- theta_itc[2,0])
n_02 = (theta_itc[1,0] - theta_itc[1,2])/(theta_itc[2,2]- theta_itc[2,0])
b_12 = (theta_itc[0,1] - theta_itc[0,2])/ (theta_itc[2,2]- theta_itc[2,1])
n_12 = (theta_itc[1,1] - theta_itc[1,2])/(theta_itc[2,2]- theta_itc[2,1])



# Predicted classes for the training set without interceptor
data_pr_0 = Xy[:,:-1][pred_cl==0]
data_pr_1 = Xy[:,:-1][pred_cl==1]
data_pr_2 = Xy[:,:-1][pred_cl==2]
# computing the the boundaries of the scores for the classses 
theta = mc.theta
m_01 = -(theta[0,0] - theta[0,1]) / (theta[1,0] - theta[1,1])
m_02 = -(theta[0,0] - theta[0,2]) / (theta[1,0] - theta[1,2])
m_12 = -(theta[0,1] - theta[0,2]) / (theta[1,1] - theta[1,2])

# Ogiginal classes for the training set
data_0 = Xy[:,:-1][Xy[:,-1]==0]
y_0 = Xy[:,-1][Xy[:,-1]==0]
data_1 = Xy[:,:-1][Xy[:,-1]==1]
y_1 = Xy[:,-1][Xy[:,-1]==1]
data_2 = Xy[:,:-1][Xy[:,-1]==2]
y_2 = Xy[:,-1][Xy[:,-1]==2]
fig, ax = plt.subplots(2, 2, figsize=(15, 15))

fig.text(0.5, 0.95, 'Without Interceptor', ha='center', fontsize=14)
ax[0,0].set_title("Original classes + predicted classification boundaries")
ax[0,0].set_ylabel("Petal Lenght (Cm)")
ax[0,0].set_xlabel("Petal Width (Cm)")
ax[0,0].plot(data_0[:,0],data_0[:,1], '.', c='r')
ax[0,0].plot(data_1[:,0],data_1[:,1], '.', c='b')
ax[0,0].plot(data_2[:,0],data_2[:,1], '.', c='m')
ax[0,0].plot(t, m_01*t)
# ax[0,0].plot(t, m_02*t)
ax[0,0].plot(t, m_12*t)


ax[0,1].set_title("Predicted classes + predicted classification boundaries")
ax[0,1].set_ylabel("Petal Lenght (Cm)")
ax[0,1].set_xlabel("Petal Width (Cm)")
ax[0,1].plot(data_pr_0[:,0],data_pr_0[:,1], '.', c='r')
ax[0,1].plot(data_pr_1[:,0],data_pr_1[:,1], '.', c='b')
ax[0,1].plot(data_pr_2[:,0],data_pr_2[:,1], '.', c='m')
ax[0,1].plot(t, m_01*t)
# ax[0,1].plot(t, m_02*t)
ax[0,1].plot(t, m_12*t)

fig.text(0.5, 0.48, 'With Interceptor', ha='center', fontsize=14)
ax[1,0].set_title("original classes + predicted classification boundaries")
ax[1,0].set_ylabel("Petal Lenght (Cm)")
ax[1,0].set_xlabel("Petal Width (Cm)")
ax[1,0].plot(data_0[:,0],data_0[:,1], '.', c='r')
ax[1,0].plot(data_1[:,0],data_1[:,1], '.', c='b')
ax[1,0].plot(data_2[:,0],data_2[:,1], '.', c='m')
ax[1,0].plot(t, b_01 + n_01*t)
# ax[1,0].plot(t, b_02 + n_02*t)
ax[1,0].plot(t, b_12 + n_12*t)

ax[1,1].set_title("Predicted classes + predicted classification boundaries")
ax[1,1].set_ylabel("Petal Lenght (Cm)")
ax[1,1].set_xlabel("Petal Width (Cm)")
ax[1,1].plot(data_itc_pr_0[:,0],data_itc_pr_0[:,1], '.', c='r')
ax[1,1].plot(data_itc_pr_1[:,0],data_itc_pr_1[:,1], '.', c='b')
ax[1,1].plot(data_itc_pr_2[:,0],data_itc_pr_2[:,1], '.', c='m')
ax[1,1].plot(t, b_01 + n_01*t)
# ax[1,1].plot(t, b_02 + n_02*t)
ax[1,1].plot(t, b_12 + n_12*t)
plt.subplots_adjust(hspace=0.6)
plt.show()
