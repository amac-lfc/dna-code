import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt


GElist = np.load("FirePkl/COADGenes.npy")
X = np.load("FirePkl/COADX.npy")
Y = np.load("FirePkl/COADY.npy")

X = np.transpose(X)
covmatrix= np.dot(X,np.transpose(X))

data = np.copy(X)
[M,N] = data.shape
mn = np.mean(data, axis=1)

rep = np.transpose(np.vstack([mn]*N))
data = data - rep
Y = np.transpose(data) / np.sqrt(N-1)

[u,S,PCT] = np.linalg.svd(Y)
PC = np.transpose(PCT)
S = np.diag(S)
V = S*S
signals = np.dot(PCT,data)

country = np.range(X.shape[1])
features = GElist
print(len(features))

print(signals[0])

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(signals[0],[0,0,0,0])
for i in range(4):
    ax.annotate(country[i], xy=(signals[0,i],0), textcoords='data')
plt.xlabel('PC1')
plt.show()


fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
plt.scatter(signals[0],signals[1])
for i in range(4):
    ax.annotate(country[i], xy=(signals[0,i],signals[1,i]), textcoords='data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

print(PC.shape)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
# PC= np.transpose(PC)
for i in range(17):
    plt.scatter(PC[i,0],PC[i,1])
    ax.annotate(features[i], xy=(PC[i,0],PC[i,1]), textcoords='data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

fig = plt.figure()
plt.clf()
plt.bar(range(len(np.diag(V))), np.diag(V), align='center', alpha=0.5)
plt.title('Wieght of each eigen value')
plt.show()
