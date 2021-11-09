import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.datasets import load_brest_cancer
cancer=load_brest_cancer()
type(cancer)

cancer.keys()

print(cancer['DESCR'])

data=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

data.head()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit()

scaled_data=scaler.transform(data)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_data)

X_pca=pca.transform(scaled_data)
scaled_data.shape
X_pca.shape

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

#Interpreting the comments(@hd)

pca.components_

data_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
