import pandas as pd
import numpy as np

MET =  pd.read_pickle("pklData/METcancer_OV_processed.pkl")
GE  = pd.read_pickle("pklData/GEcancer_OV_processed.pkl")

print(MET, GE)

X = np.load("pklData/Matrix_X.npy")
Y = np.load("pklData/Matrix_Y.npy")
newY = np.load("pklData/Matrix_NewY.npy")
interface = np.load("pklData/Interface.npy")

print(X)
print(Y)
print(newY)
print(interface)
