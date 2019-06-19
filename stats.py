import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Y = np.load("FirePkl/Y.npy")
X = np.load("FirePkl/X.npy")
GEN = np.load("FirePkl/GEN.npy")
GE = np.load("FirePkl/GE.npy")
GElist = np.load("FirePkl/GElist.npy")

#plots
f = np.arange(len(X[0]), dtype=int)
f = np.arange(len(X[1]), dtype = int)

XmGEN = np.abs(np.mean(X, 0) - GEN)

p90 = np.percentile(XmGEN, 90)
print(p90)

indexp90 = np.where(XmGEN >= p90 )


print(GElist[indexp90])
print(len(GElist[indexp90]))

# plt.figure(1)
# plt.scatter(f, XmGEN)

# plt.figure(2)
# plt.scatter(f, GEN)

# plt.show()