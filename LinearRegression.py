import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = np.load("pklData/Matrix_X.npy")
Y = np.load("pklData/Matrix_NewY.npy")
# H = np.load("pklData/Matrix_H.npy")

X = np.transpose(X)
Y = np.transpose(Y)
# H = np.transpose(H)

# X2 = np.hstack((X, H))

mlr = LinearRegression()
mlr.fit(X, Y)

Y1 = np.load("pklData/Matrix_Y.npy")
# print(mlr.coef_)
# print(mlr.intercept_)


#polynmial Regression
model = Pipeline([('poly', PolynomialFeatures(degree=2)),\
        ('linear', LinearRegression(fit_intercept=False))])
# fit to an order-2 polynomial data
model = model.fit(X[0, :][0], Y[0])
print(model.named_steps['linear'].coef_)


# #print statment
#
#
# i = 1000
# for j in range(i):
#     plt.figure()
#     plt.plot(Poly)
#     plt.scatter(X[j], Y[j])
#     plt.savefig("MatPlotFigs/"+str(j))
# #plt.show()
#


# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# i=0
#
# ax.plot_trisurf(X[i], Y2[i], linewidth=0.2, antialiased=True)
# ax.scatter(X[i, 0], H[i, 0], Y[i, 0], c="r", marker="^")
#
# ax.set_xlabel("Methelation")
# ax.set_ylabel('Hypo/Hyper Methelation')
# ax.set_zlabel("Expression")
#
# plt.show()
#



# NOT IN USE
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Grab some test data.
# X, Y, Z = axes3d.get_test_data(0.05)
#
# print(X.shape)
# print(Y.shape)
# print(Z.shape)
# X, H = np.meshgrid(X, H, indexing = "ij")
# Plot a basic wireframe.
# ax.plot_wireframe(X, H, Y2, rstride=10, cstride=10)
#
# plt.show()
