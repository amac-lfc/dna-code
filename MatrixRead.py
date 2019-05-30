import pandas as pd
import numpy as np

MET =  pd.read_pickle("METcancer_OV_processed.pkl")
GE  = pd.read_pickle("GEcancer_OV_processed.pkl")

print(MET, GE)

X = np.load("Matrix_X.pkl.npy")
Y = np.load("Matrix_Y.pkl.npy")
interface = np.load("Interface.pkl.npy")
print(X)
print(Y)
print(interface)
