import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

Y = np.load("FirePkl/Y.npy")
X = np.load("FirePkl/X.npy")

print(Y.shape)
print(X.shape)

PCAresult = PCA()
fit = PCAresult.fit(X,y = Y)

print(PCAresult.explained_variance_ratio_)

print(PCAresult.components_)