import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

GElist = np.load("Pickles/COADGenes.npy")
X = np.load("Pickles/COADX.npy")
Y = np.load("Pickles/COADY.npy")
Y = Y.astype(int)

print(X)

pca = PCA(n_components=100)
Components = pca.fit_transform(X)

aliveIndex = Y > 0


# print(principalDf)
alive = np.where(aliveIndex == True)[0]
dead = np.where(aliveIndex == False)[0]

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

X_ori = pca.inverse_transform(Components[:, 0])
print(X_ori)
