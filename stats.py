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

plt.figure(1)
plt.scatter(f, X[0])

plt.figure(2)
plt.scatter(f, GEN)

plt.show()