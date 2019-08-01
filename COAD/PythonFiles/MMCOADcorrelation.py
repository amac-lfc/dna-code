import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

ExpMean = np.load('FirePkl/MMCOADExpMean.npy')
ExpPatients = np.load('FirePkl/MMCOADExpPatients.npy')
ExpValues = np.load('FirePkl/MMCOADExpValues.npy')
ExpGenes = np.load('FirePkl/MMCOADExpGenes.npy')

MethylGenes = np.load('FirePkl/MMCOADMethylGenes.npy')
MethylMean = np.load('FirePkl/MMCOADMethylMean.npy')
MethylPatients = np.load('FirePkl/MMCOADMethylPatients.npy')
MethylValues = np.load('FirePkl/MMCOADMethylValues.npy')

MethylPatientsCutOff = np.empty(len(MethylPatients), dtype=object)
for i in range(len(MethylPatients)):
    MethylPatientsCutOff[i] = str(MethylPatients[i][:15])

Bpat = np.empty((len(MethylPatientsCutOff), 2), dtype=int)
c = 0
for i in range(len(MethylPatientsCutOff)):
    for j in range(len(ExpPatients)):
        if str(MethylPatientsCutOff[i]) == str(ExpPatients[j]):
            c +=1
            Bpat[i] = [i, j]

index = np.where(Bpat < 10000)
Bpat = Bpat[index]
Bpat = Bpat.reshape(175,2)

X = MethylValues[Bpat[:,0]]
Y = ExpValues[Bpat[:,0]]

Bpat = 0
print(X.shape)
print(Y.shape)

for i in range(ExpGenes.shape[0]):
    ExpGenes[i] = ExpGenes[i].split('|')[0]

# print(ExpGenes)
# print(MethylGenes)
BGenes = np.zeros((len(ExpGenes), 2))
c = 0
for i in range(len(ExpGenes)):
    for j in range(len(MethylGenes)):
        if str(ExpGenes[i]) == str(MethylGenes[j]):
            c +=1
            BGenes[i, 0] = i
            BGenes[i, 1] = j
            

# BGenes = np.nonzero(BGenes)
BGenes = np.column_stack((BGenes[0], BGenes[1]))    
print(c)
print(BGenes)   
print(BGenes.shape)

X = X[BGenes[:, 0], :]
Y = Y[BGenes[:, 1], :]
print(X.shape)
print(Y.shape)

for i in range(len(BGenes)):
    plt.ylabel("Gene Expression")
    plt.xlabel("Methylation")
    plt.xlim(-1,1)
    plt.ylim(-1, 1)
    plt.scatter(X[i], Y[i])
    plt.savefig("MatPlotFigsMMCOAD/"+str(i), bbox_inches = 'tight')

print('it should be saved')
# # degree
# degree = 3

# Xf = X.flatten()
# Yf = Y.flatten()

# #polynmial Regression
# model = Pipeline([('poly', PolynomialFeatures(degree=degree)),\
#         ('linear', LinearRegression(fit_intercept=False))])

# # fit to an order-2 polynomial data
# model = model.fit(Xf[:, np.newaxis], Yf)
# coefs =  model.named_steps['linear'].coef_

# X2 = np.linspace(0.0, 1.0, 100)
# Y2 = np.zeros(len(X2),'d')+coefs[0]
# for i in range(1, len(coefs)):
#     Y2 += X2**i*coefs[i]

# N = Bpat.shape[0]
# plt.figure()
# for j in range(N):
#     plt.clf()
#     plt.ylabel("Gene Expression")
#     plt.xlabel("Methelation")
#     plt.xlim(0,1)
#     plt.ylim(-3, 4)
#     plt.scatter(X[j], Y[j])
#     plt.plot(X2, Y2, 'r')
#     plt.savefig("MatPlotFigs/"+str(j))

# plt.figure()
# plt.scatter(X, Y)
# plt.plot(X2, Y2)
# plt.show()
