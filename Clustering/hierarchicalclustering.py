import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("Mall_Customers.csv")
data.head()

import scipy.cluster.hierarchy as sch
dendogram=sch.dendogram(sch.linkage(X,method='ward'))
plt.title("Dendogram")
plt.xlabel('customers')
plt.ylabel('Euclidean Distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_ch=hc.fit_predict(X)