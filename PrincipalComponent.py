import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

GElist = np.load("FirePkl/COADGenes.npy")
X = np.load("FirePkl/COADX.npy")
Y = np.load("FirePkl/COADY.npy")

pca = PCA(n_components=100)
Components = pca.fit_transform(X)

# principalDf = pd.DataFrame(data = Components, columns = ['principal component 1', 'principal component 2'])

# print(principalDf)

plt.scatter(Components[:, 0], Components[:, 1])
plt.show()

X_ori = pca.inverse_transform(Components[:, 0])
print(X_ori)
