import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("50_Stratups.csv")
data.head()

X=data.iloc[:,:-1].values
Y=data.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder()
a=onehotencoder.fit_transform(X[:,[3]]).toarray()

##avoiding Dummy variable
a=a[:,1:]

X=X[:,:3]
X=np.concatenate((X,a),axis=1)

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(Xtrain,Ytrain)

y_pred=regressor.predict(Xtest)

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

X_opt=np.array(X[:,[0,1,2,3]],dtype = float)
regressor_OLS=sm1.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

#forward selection
X_opt2=np.array(X[:,[0]],dtype = float)
regressor_OLS2=sm1.OLS(endog=y,exog=X_opt2).fit()
regressor_OLS2.summary()



