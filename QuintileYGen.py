import numpy as np
import pandas as pd

Y = np.load("FirePkl/Y.npy")
X = np.load("FirePkl/X.npy")
Y = Y.astype(int)
index = Y > 0
YD = Y[index]
XD = X[index, :]

np.save("YD.npy", YD, )

YPercentiles = np.percentile(YD, [20, 40, 60, 80])

for i in range(len(YD)):
    if YD[i] < YPercentiles[0]:
        YD[i] = 1
    elif YD[i] < YPercentiles[1]:
        YD[i] = 2
    elif YD[i] < YPercentiles[2]:
        YD[i] = 3
    elif YD[i] < YPercentiles[3]:
        YD[i] = 4
    else:
        YD[i] = 5

YD = YD.astype(int)

np.save("FirePkl/YQuintiles", YD, allow_pickle= True)
np.save("FirePkl/XDead", XD, allow_pickle=True)