import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklean.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns

data=pd.read_csv("Social_Network_Ads.csv")
data.head()

data.isnull().sum()

X=data.iloc[:,[2,3]].values
Y=data.iloc[:,4].values

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.20,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Xtrain=sc.fit_transform(Xtrain)
Xtest=sc.fit_transform(Xtest)

from sklearn.svm import SVC
classifier=SVC(kernal='rbf',random_state=0)
classifier.fit(Xtrain,Ytrain)

y_pred=classifier.fit(Xtest)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Ytest,y_pred)