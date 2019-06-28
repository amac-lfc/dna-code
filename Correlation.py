import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as sc

Y = np.load("FirePkl/Y.npy")
X = np.load("FirePkl/X.npy")

X = X.astype(float)
Y = Y.astype(int)

Y2CoreSig = np.zeros((len(X[1, :]), 2))
for i in range(len(X[1, :])):
    Y2CoreSig[i] = sc.pearsonr(X[:, i],Y)
    # print(sc.pearsonr(X[:, i],Y2))
    # print(Y2CoreSig[i])

np.save("FirePkl/Y2CoreSig", Y2CoreSig, allow_pickle=True)
print(Y2CoreSig)

