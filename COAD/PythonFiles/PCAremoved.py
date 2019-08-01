import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

GElist = np.load("FirePkl/COADGenes.npy")
X = np.load("FirePkl/COADX.npy")
Y = np.load("FirePkl/COADY.npy")

pca = PCA(n_components=100)
Components = pca.fit_transform(X)

# principalDf = pd.DataFrame(data = Components, columns = ['principal component 1', 'principal component 2'])


# print(principalDf)
keeplist = np.where(Components[:,0] < 60)[0]
print(keeplist)
Components = Components[keeplist,:]
print(Components.shape) 
Y = Y[keeplist]
GElist = GElist[keeplist]
np.save('FirePkl/COAD_OutlierRemovedX', Components, allow_pickle=True)
np.save('FirePkl/COAD_OutlierRemovedY', Y, allow_pickle=True)
np.save('FirePkl/COAD_OutlierRemovedGenes', GElist, allow_pickle=True)

alive = np.where(Y == 1)[0]
dead = np.where(Y == 0)[0]


plt.scatter(Components[dead, 0], Components[dead, 1], color= 'r')
plt.scatter(Components[alive, 0], Components[alive, 1], color = 'b')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x =Components[dead, 0]
y =Components[dead, 1]  
z =Components[dead, 2]
ax.scatter(x, y, z, c='r', marker='o')
x =Components[alive, 0]
y =Components[alive, 1]
z =Components[alive, 2]
ax.scatter(x, y, z, c='b', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


