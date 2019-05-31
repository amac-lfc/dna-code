import pandas as pd
import numpy as np
import scipy.stats as sy

# load the CSV and get all gene names
MET =  pd.read_pickle("pklData/METcancer_OV_processed.pkl")
GE  = pd.read_pickle("pklData/GEcancer_OV_processed.pkl")
gL = MET["genes"].values

# load the numpy matrixes, interface,
# and make a new Y thats in the same order as X
X = np.load("pklData/Matrix_X.npy")
Y = np.load("pklData/Matrix_Y.npy")
interface = np.load("pklData/Interface.npy")
newY = Y[interface]

# correlation creation
Cor = np.zeros(0)
sig = np.zeros(0)
for i in range(X.shape[0]):
    temp = sy.stats.pearsonr(X[i][:], newY[i][:])
    Cor = np.append(Cor, temp[0])
    sig = np.append(sig, temp[1])

# adding gene labels to the correlation array and writing
t = np.array([Cor, sig, gL])
set = pd.DataFrame({'Gene':gL[:],'Correlation':Cor[:], 'Significance':sig[:]})

set = set.sort_values("Significance", ascending = True)
v = set['Significance'].values[:200]

set.to_csv("Correlation.csv", index = True)

#first N siginifcant
ind = set.index.tolist()
N = 200
X2 = X[ind[:200]]
Y2 = newY[ind[:200]]
np.save("pklData/Matrix_X2", X2, allow_pickle = True)
np.save("pklData/Matrix_Y2", Y2, allow_pickle = True)
