##K-Nearest Neighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(Xtrain,Ytrain)

##SVM

from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(Xtrain,Ytrain)

##kernel Svm

from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(Xtrain,Ytrain)


