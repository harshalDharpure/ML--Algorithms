import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("Social_Network_Ads.csv")
data.head()

X=data.iloc[:,[2,3]].values
Y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.30,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Xtrain=sc.fit_transform(Xtrain)
Ytrain=sc.fit_transform(Ytrain)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(Xtrain,Ytrain)

y_pred=classifier.predict(Xtest)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Ytest,y_pred)


##prediction and Evaluation

predictions =dtree.predict(Xtest)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Ytest,predictions))
print(confusion_matrix(Ytest,predictions))

