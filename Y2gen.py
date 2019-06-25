import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as sc

Y = np.load("FirePkl/Y.npy")
X = np.load("FirePkl/X.npy")

Y2 = np.zeros(len(Y))
Y = Y.astype(int)

for i in range(len(Y)):
    if Y[i] < 0:
        Y2[i] = 7 
    elif Y[i] < 501:
        Y2[i] = 1
    elif Y[i] < 1001:
        Y2[i] = 2
    elif Y[i] < 1501:
        Y2[i] = 3
    elif Y[i] < 2001:
        Y2[i] = 4
    elif Y[i] < 2501:
        Y2[i] = 5
    elif Y[i] > 3000:
        Y2[i] = 6


reg = np.zeros(len(X[:,1]), dtype = object)
for i in range(len(X[:,1])):
    reg[i] = sc.pearsonr(X[:,i],Y)

reg = reg[np.argsort(reg[:,1][1])]
print(reg)