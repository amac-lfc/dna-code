import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats

ExpGenes = np.load('FirePkl/OVExpGenes.npy')
ExpPatients = np.load('FirePkl/OVExpPatients.npy')
ExpNumpy = np.load('FirePkl/OVExpValues.npy')
ExpMean = np.load('FirePkl/OVExpMean.npy')

MethylGenes = np.load('FirePkl/OVMethylGenes.npy')
MethylPatients = np.load('FirePkl/OVMethylPatients.npy')
MethylValues = np.load('FirePkl/OVMethylValues.npy')
MethylMean = np.load('FirePkl/OvMethylMean.npy')

BothGenesExp = np.isin(ExpGenes, MethylGenes)
BothGenesMethyl = np.isin(MethylGenes, ExpGenes)
EBGenes = ExpGenes[BothGenesExp]
MBGenes = MethylGenes[BothGenesMethyl]
BothGenesIndex = np.empty(len(EBGenes))
for i in range(len(EBGenes)):
    BothGenesIndex[i] = np.where(EBGenes[i]== MethylGenes)[0][0]
BothGenesIndex = BothGenesIndex.astype(int)

# degree
degree = 2

X = ExpMean
Y = MethylMean[BothGenesIndex[:]]

#polynmial Regression
# model = Pipeline([('poly', PolynomialFeatures(degree=degree)),\
#         ('linear', LinearRegression(fit_intercept=False))])

# fit to an order-2 polynomial data
# model = model.fit(Xf[:, np.newaxis], Yf)
# coefs =  model.named_steps['linear'].coef_

# X2 = np.linspace(0.0, 1.0, 100)
# Y2 = np.zeros(len(X2),'d')+coefs[0]
# for i in range(1, len(coefs)):
#     Y2 += X2**i*coefs[i]

N = 5000 
plt.figure()
plt.clf()
for j in range(N):
    plt.title("Methelation and Expression")
    plt.ylabel("Methylation")
    plt.xlabel("Gene Expression")
    plt.xlim(2,13)
    plt.ylim(0, 1)
    plt.scatter(X[j], Y[j])
    # plt.plot(X2, Y2, 'r')

# slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
# line = slope*X+intercept
# plt.plot(line)
plt.savefig("MatPlotFigsOV/"+str(j), bbox_boarder = 'tight')
# plt.figure()
# plt.scatter(X, Y)
# plt.plot(X2, Y2)
# plt.show()
