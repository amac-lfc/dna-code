import numpy as np
import numpy.core.defchararray as np_f
import pandas as pd
import matplotlib.pyplot as plt

Y = np.load("FirePkl/Y.npy")
Y = Y.astype(int)
X = np.load("FirePkl/X.npy")
X = X.astype(int)
RF = np.loadtxt("TotalImportanceTopBinary.txt", dtype=str)
GElist = np.load("FirePkl/GElist.npy")
NonBinary = np.load("FirePkl/TotalImportanceTopNonBinary.npy")
NonBinaryInterface = np.load("FirePkl/RFTopInterface.npy")

NonBinary0 = NonBinary[:, 0]
NonBinary1 = NonBinary[:,1]
NonBinary1 = NonBinary1.astype(float)

RF = np_f.replace(RF, "['", "")
RF = np_f.replace(RF, '["', "")
RF = np_f.replace(RF, "]", "")
RF2 = [x[:-1] for x in RF[:,0]]
RF3 = RF[:, 1]
RF3 = RF3.astype(float)

index0 = Y == -1
index1 = Y > 0

top = 10

Interface = np.zeros(len(RF)-3)
k=0
i=0
while i < (len(RF2)-3):
        if (RF2[k] == 'Treatment' or RF2[k] == 'Age' or RF2[k] == 'Stage'):
                k = k+1
        else:
                temp = (np.where(RF2[k] == GElist)[0])
                Interface[i] = temp[0]
                k += 1
                i += 1
Interface = Interface.astype(int)

DX = X[index0, :]
LX = X[index1, :]

LAllAdvs = LX.sum(axis = 0)
LGeneAdvs = LAllAdvs[Interface[:top]]
LGeneAdvs = [x / LX.shape[0] for x in LGeneAdvs]

DAllAdvs = X.sum(axis = 0)
DGeneAdvs = DAllAdvs[Interface[:top]]
DGeneAdvs = [x / DX.shape[0] for x in DGeneAdvs]

GeneNames = GElist[Interface]
t= np.arange(top)

plt.bar(t, height = DGeneAdvs)
plt.ylabel('Mean Expression')
plt.xticks(t, GeneNames)
plt.title("Average Expression For the Top top Gene Predictors (Deceased Patients)")
plt.savefig("Report/figure1.eps", bbox_inches='tight')

plt.clf()

plt.bar(t, height = LGeneAdvs)
plt.ylabel('Mean Expression')
plt.xticks(t, GeneNames)
plt.title("Average Expression For the Top top Gene Predictors (Alive Patients)")
plt.savefig("Report/figure2.eps", bbox_inches='tight')

plt.clf()

plt.bar(t, height = RF3[:top])
plt.ylabel('Factor Weight')
plt.xticks(t, GeneNames)
plt.title("Average Weight of Each Gene on Prediction of Vital Status")
plt.savefig("Report/figure3.eps", bbox_inches='tight')

plt.clf()


plt.bar(t, height = NonBinary1[:10])
plt.ylabel('Factor Weight')
plt.xticks(t, NonBinary0[:10])
plt.title("Average Weight of Each Gene on Prediction of Days to Death")
plt.savefig("Report/figure4.eps", bbox_inches='tight')