import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data=pd.read_csv("Salary_Data.csv")
data.head()

from sklearn.cross_validation import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.30,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(Xtrain,Ytrain)


y_pred=regressor.predict(Xtest)

##training set result

plt.scatter(Xtrain,Ytrain,color='red')
plt.plot(Xtrain,regressor.predict(Xtrain),color='blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel('Years of experience')
plt.ylabel("salary")
plt.show()

#for testing set result

plt.scatter(Xtest,Ytest,color='red')
plt.plot(Xtest,regressor.predict(Xtest),color='blue')
plt.title('salary vs experience(Test Ste)')
plt.xlabel("Years of experience")
plt.ylabel("salary")
plt.show()

##visualizing the column

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#regressor.fit


print(regressor.intercept_)
print(regressor.coef_)
data.columns

sns.pairplot(data)

sns.displot(data["selling price"])

sns.heatmap(data.corr(),annot=True)

#predictions

predictions=regressor.predict(Xtest)
predictions

plt.scatter(Ytest,predictions)
sns.displot((Ytest-predictions))

from sklearn import metrics
metrics.mean_absolute_error(Ytest,predictions)
metrics.mean_squared_error(Ytest,predictions)
np.sqrt(metrics.mean_squared_error(Ytest,predictions))

