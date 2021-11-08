import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("Social_Network.Ads.csv")
data.head()

X=data.iloc[:,[2,3]].values
Y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Xtrain=sc.fit_transform(Xtrain)
Xtest=sc.fit_transform(Xtest)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(Xtrain,Ytrain)

y_pred=classifier.predict(Xtest)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Ytest,y_pred)

import matplotlib.pyplot as plt
%matplotlib inline

##Exploratory Data Analysis

sns.pairplot(data,hue='Social_Network',palette='set1')


##prediction and Evaluation

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Ytest,y_pred))

print(confusion_matrix(Ytest,y_pred))

