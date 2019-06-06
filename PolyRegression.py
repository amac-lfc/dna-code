import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = np.load("pklData/Matrix_X.npy")
Y = np.load("pklData/Matrix_NewY.npy")


X = X[0]
Y = Y[0]

#polynmial Regression
model = Pipeline([('poly', PolynomialFeatures(degree=2)),\
        ('linear', LinearRegression(fit_intercept=False))])

# fit to an order-2 polynomial data
model = model.fit(X[:, np.newaxis], Y)
print(model.named_steps['linear'].coef_)