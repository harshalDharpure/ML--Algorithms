import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("Social_Network_Ads.csv")
X=data.iloc[:,[2,3]].values
Y=data.iloc[:,4].values

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.20,random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Xtrain=sc.fit_transform(Xtrain)
Xtest=sc.fit_transform(Xtest)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(Xtrain,Xtest)

y_pred=classifier.predict(Xtest)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix()

import seaborn as sns
sns.heatmap(cm,cmap='plasma',fmt='d',annot=True) 


##prediction and Evaluation

from sklearn.metrics import classification_report
print(classification_report(Ytest,y_pred))

       