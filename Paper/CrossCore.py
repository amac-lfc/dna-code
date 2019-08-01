import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy.stats as sp 
training = np.load("PapersDataSet/sample_training_replacement_mean.npy")
training = training[:,:-1]
testing = np.load("PapersDataSet/sample_testing_replacement_mean.npy")
features = np.load("PapersDataSet/sample_features.npy")
print(features)

CCMatrix = np.empty((training.shape[1], training.shape[1]))
for i in range(training.shape[1]):
    for k in range(training.shape[1]):
        CCMatrix[i, k] =  (sp.pearsonr(training[:, i], training[:, k]))[0]

np.savetxt("PapersDataSet/CrossCorelationMatrix", CCMatrix, delimiter='\t')

print(CCMatrix.shape)

x = range(67)
y = range(67)

xx, yy = np.meshgrid(x, y)
plt.contourf(xx, yy, CCMatrix)
plt.colorbar()
plt.savefig("CrossCorrelatonHeatMap", bbox_inches='tight')