import numpy as np
import pandas as pd
import scipy.stats as sy 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

Y = np.load("FirePkl/Y.npy")
X = np.load("FirePkl/X.npy")
GEN = np.load("FirePkl/GEN.npy")
GE = np.load("FirePkl/GE.npy")
GElist = np.load("FirePkl/GElist.npy")

#need to change Y data type to int
Y = Y.astype(np.float)

#plots
f = np.arange(len(X[0]), dtype=int)
f = np.arange(len(Y), dtype = int)

XmGEN = np.abs(np.mean(X, 0) - GEN)
p90 = np.percentile(XmGEN, 90)
indexp90 = np.where(XmGEN >= p90 )

XCoreList = np.zeros((len(X[:,0])))
for i in range(len(X[:,0])):
    gene = list(X[:,i])
    patient = list(Y[:])
    XCoreList[i] = sy.stats.pearsonr(patient, gene)[0]
print(XCoreList)
XCoreList = np.sort(XCoreList)
np.savetxt("CorrelationsForFireBrowsep90.txt", XCoreList)
 

tenHigh = XCoreList.argsort()[-10:][::-1]
print(GElist[tenHigh])

index = np.where(XCoreList == np.amax(XCoreList))
print(GElist[index])
print(GEN[index])


plt.hist(Y, bins=[-100, 0, 200, 400, 600, 800, 1000])


plt.show()