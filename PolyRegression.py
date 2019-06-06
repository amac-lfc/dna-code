import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = np.load("pklData/Matrix_X.npy")
Y = np.load("pklData/Matrix_NewY.npy")


# Gene :
i = 1000

# degree
degree = 3

X = X[i]
Y = Y[i]

#polynmial Regression
model = Pipeline([('poly', PolynomialFeatures(degree=degree)),\
        ('linear', LinearRegression(fit_intercept=False))])

# fit to an order-2 polynomial data
model = model.fit(X[:, np.newaxis], Y)
coefs =  model.named_steps['linear'].coef_

X2 = np.sort(X, axis=None)
Y2 = np.zeros(len(X2),'d')+coefs[0]
for i in range(1, len(coefs)):
    Y2 += X2**i*coefs[i]

plt.figure()
plt.scatter(X, Y)
plt.plot(X2, Y2)
plt.show()