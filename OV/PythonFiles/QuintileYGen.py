import numpy as np
import pandas as pd

Y = np.load("FirePkl/COAD_OutlierRemovedY.npy")
X = np.load("FirePkl/X.npy")
Y = Y.astype(int)
index = Y > 0
Y = Y[index]
YD = np.copy(Y)
# XD = X[index, :]

np.save("YD.npy", YD, )

YPercentiles = np.percentile(YD, [20, 40, 60, 80])

print(YPercentiles)
for i in range(len(YD)):
    if Y[i] < YPercentiles[0]:
        YD[i] = 1
    elif Y[i] < YPercentiles[1]:
        YD[i] = 2
    elif Y[i] < YPercentiles[2]:
        YD[i] = 3
    elif Y[i] < YPercentiles[3]:
        YD[i] = 4
    else:
        YD[i] = 5

YD = YD.astype(int)

newY = np.zeros((len(YD), 5), dtype=int)
for i in range(len(YD)):
    if Y[i] < YPercentiles[0]:
        newY[i, 0] = 1
    elif Y[i] < YPercentiles[1]:
        newY[i, 1] = 1
    elif Y[i] < YPercentiles[2]:
        newY[i, 2] = 1
    elif Y[i] < YPercentiles[3]:
        newY[i, 3] = 1
    else:
        newY[i, 4] = 1

print(Y)
np.save("FirePkl/COADYQuintiles_array", newY, allow_pickle=True)
np.save("FirePkl/COADYQuintiles", YD, allow_pickle= True)
# np.save("FirePkl/XDead", XD, allow_pickle=True)