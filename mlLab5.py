from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import random

col = [
    "duration",
    "total_fiat",
    "total_biat",
    "min_fiat",
    "min_biat",
    "max_fiat",
    "max_biat",
    "mean_fiat",
    "mean_biat",
    "flowPktsPerSecond",
    "flowBytesPerSecond",
    "min_flowiat",
    "max_flowiat",
    "mean_flowiat",
    "std_flowiat",
    "min_active",
    "mean_active",
    "max_active",
    "std_active",
    "min_idle",
    "mean_idle",
    "max_idle",
    "std_idle",
    "class1"
]
# df=pd.read_csv("ml_lab1/lab3/lab3_vpn_nonvpndataset.txt",names=col)
df=pd.read_csv(r"C:\Users\dhanu\OneDrive\Desktop\jup\ml_lab1\2016_dataset\combine.txt",names=col)

print("shape of df before removing nan values",df.shape)
df=df.dropna()
print("shape of df after removing nan values",df.shape)

#

X,y=df.loc[:,["min_idle"]],df["std_idle"]


xtrain,xtest,ytrain,ytest=train_test_split (X, y, test_size=0.33, random_state=42,shuffle=True)


reg =  LinearRegression().fit(xtrain, ytrain)
y_test_pred = reg.predict(xtest)
print(len(ytest))
print(len(y_test_pred))


#a2
def evaluate_price_prediction(actual, predicted): # Function to calculate metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mse, rmse, mape, r2

mse,rmse,mape,r2=evaluate_price_prediction(ytest,y_test_pred)

print(mse,rmse,r2,mape)

#a3
A,b=df.loc[:,["min_idle","mean_idle","max_idle"]],df["std_idle"]
xtrain,xtest,ytrain,ytest=train_test_split (A, b, test_size=0.33, random_state=42,shuffle=True)


reg =  LinearRegression().fit(xtrain, ytrain)
y_test_pred = reg.predict(xtest)
print(len(ytest))
print(len(y_test_pred))


def evaluate_price_prediction(actual, predicted): # Function to calculate metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mse, rmse, mape, r2

mse,rmse,mape,r2=evaluate_price_prediction(ytest,y_test_pred)

print(mse,rmse,r2,mape)

#a4
from sklearn.cluster import KMeans

attr=df.loc[:,"duration":"std_idle" ]
print(attr)

kmeans = KMeans(n_clusters=2, random_state=0,n_init="auto").fit(attr)
print(kmeans.labels_)
print(kmeans.cluster_centers_)


from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

kmeans = KMeans(n_clusters=2, random_state=42).fit(attr)
a=silhouette_score(attr, kmeans.labels_)
c=calinski_harabasz_score(attr, kmeans.labels_)
d=davies_bouldin_score(attr, kmeans.labels_)
print(a,c,d)

#a6
sil_score=list()
cal_score=list()
davi_score=list()
for i in range(2,7):
  kmeans=KMeans(n_clusters=i,random_state=42).fit(attr)
  a=silhouette_score(attr, kmeans.labels_)
  sil_score.append(a)
  c=calinski_harabasz_score(attr, kmeans.labels_)
  cal_score.append(c)
  d=davies_bouldin_score(attr, kmeans.labels_)
  davi_score.append(d)

x=np.array(range(2,7))
plt.plot(x,sil_score)
plt.plot(x,cal_score)
plt.plot(x,davi_score)

#a7
distorsions=list()
for k in range(2, 20):
   kmeans = KMeans(n_clusters=k).fit(attr)
   distorsions.append(kmeans.inertia_)
plt.plot(distorsions)