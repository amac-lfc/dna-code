import pandas as pd
import numpy as np

MET =  pd.read_pickle("pklData/METcancer_OV_processed.pkl")
GE  = pd.read_pickle("pklData/GEcancer_OV_processed.pkl")

print(MET, GE)

X = np.load("pklData/Matrix_X.pkl.npy")
Y = np.load("pklData/Matrix_Y.pkl.npy")
interface = np.load("pklData/Interface.pkl.npy")
newY = Y[interface]

print(X)
print(Y)
print(newY)
print(interface)
