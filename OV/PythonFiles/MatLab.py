import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt

X = [[105, 103, 103, 66], [245, 227, 242, 267], [685, 803, 750, 586], \
    [147, 160, 122, 93], [193, 235, 184, 209], [156, 175, 147, 139], \
    [720, 874 ,566 ,1033],[253, 265, 171 ,143], [488, 570 ,418 ,355], \
    [198, 203, 220, 187], [360, 365, 337, 334], [1102, 1137, 957, 654],\
    [1472, 1582, 1462, 1494], [57, 73, 53, 47], [1374, 1256, 1572, 1506],\
    [375, 475, 458, 135], [54, 64, 62, 41]]
X = np.array(X)
covmatrix= np.dot(X,np.transpose(X))
data = np.copy(X)
[M,N] = data.shape
mn = np.mean(data, axis=1)
print(mn)
print(data)
rep = np.transpose(np.vstack([mn]*N))
print(rep)
print(rep.shape)
data = data - rep
print(data)
Y = np.transpose(data) / np.sqrt(N-1)
[u,S,PCT] = np.linalg.svd(Y)
PC = np.transpose(PCT)
S = np.diag(S)
V = S*S
signals = np.dot(PCT,data)

country = ['England', 'Wales', 'Scotland', 'N Ireland']
features = ['Cheese', 'Carcass mneat', 'other meat', 'Fish', 'Fats and oils', 'Sugars', \
'Fresh potatoes', 'Fresh Veg', 'Other Veg', 'Processed potatoes', 'Processe Veg', 'Fresh Fruit', 'Cereals', \
'Beverages', 'Soft Drinks', 'Alcoholic drings', 'Confectionery']

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
