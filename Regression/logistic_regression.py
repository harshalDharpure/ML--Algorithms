import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("Social_Network.Ads")
data

X=data.iloc[:,2:4].values
Y=data.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
Xtrain=sc.fit_transform(Xtrain)
Xtest=sc.fit_transform(Xtest)


from skealrn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(Xtrain,Ytrain)

y_pred=classifier.predict(Xtest)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Ytest,y_pred)

from sklearn.metrics import classification_report
print(classification_report(Ytest,prediction))



##Exploratory Data Analysis

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survied',data,palette='RdBu_r')
sns.countplot(x='Survied',hue='Sex',data,palette='RdBu_r')
sns.countplot(x='Survied',hue='pclass',data,palette='rainbow')

##Displot
sns.displot(data['Age'].dropna(),kde=False,color='darkred',bins=30)
data['age'].hist(bins=30,color='darkred',alpha=0.7)

##classification report


from sklearn.metrics import classification_report
print(classification_report(Ytest,prediction))
