import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = np.load("pklData/Matrix_X.npy")
Y = np.load("pklData/Matrix_NewY.npy")

X = np.transpose(X)
Y = np.transpose(Y)


N = 1000
for j in range(N):
    plt.figure()
    plt.scatter(X[j], Y[j])
    plt.savefig("MatPlotFigs/"+str(j))
