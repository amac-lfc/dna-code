import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = np.load("pklData/Matrix_X.npy")
Y = np.load("pklData/Matrix_NewY.npy")
MET = pd.read_pickle("pklData/METcancer_OV_processed.pkl")

# degree
degree = 3

Xf = X.flatten()
Yf = Y.flatten()

#polynmial Regression
model = Pipeline([('poly', PolynomialFeatures(degree=degree)),\
        ('linear', LinearRegression(fit_intercept=False))])

# fit to an order-2 polynomial data
model = model.fit(Xf[:, np.newaxis], Yf)
coefs =  model.named_steps['linear'].coef_

X2 = np.linspace(0.0, 1.0, 100)
Y2 = np.zeros(len(X2),'d')+coefs[0]
for i in range(1, len(coefs)):
    Y2 += X2**i*coefs[i]

N = 5000 
plt.figure()
for j in range(N):
    plt.clf()
    plt.title("" + MET["genes"].values[j])
    plt.ylabel("Gene Expression")
    plt.xlabel("Methelation")
    plt.xlim(0,1)
    plt.ylim(-3, 4)
    plt.scatter(X[j], Y[j])
    plt.plot(X2, Y2, 'r')
    plt.savefig("MatPlotFigs/"+str(j))

# plt.figure()
# plt.scatter(X, Y)
# plt.plot(X2, Y2)
# plt.show()
