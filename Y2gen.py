import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as sc

Y = np.load("FirePkl/Y.npy")
X = np.load("FirePkl/X.npy")

Y2 = np.zeros(len(Y), dtype = int)
Y = Y.astype(int)

for i in range(len(Y)):
    if Y[i] < 0:
        Y2[i] = 6
    elif Y[i] < 501:
        Y2[i] = 0
    elif Y[i] < 1001:
        Y2[i] = 1
    elif Y[i] < 1501:
        Y2[i] = 2
    elif Y[i] < 2001:
        Y2[i] = 3
    elif Y[i] < 2501:
        Y2[i] = 4
    elif Y[i] > 3000:
        Y2[i] = 5

print(Y2)
np.save("FirePkl/Y2", Y2, allow_pickle= True)

X = X.astype(int)

print(len(Y2))
print(len(Y2[Y2==0]))
print(len(Y2[Y2==1]))

Y2CoreSig = np.zeros((len(X[1, :]), 2))
for i in range(len(X[1, :])):
    Y2CoreSig[i] = sc.pearsonr(X[:, i],Y2)
    # print(sc.pearsonr(X[:, i],Y2))
    # print(Y2CoreSig[i])

np.save("FirePkl/Y2CoreSig", Y2CoreSig, allow_pickle=True)
print(Y2CoreSig)

