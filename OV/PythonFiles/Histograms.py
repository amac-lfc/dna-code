import numpy as np
import numpy.core.defchararray as np_f
import pandas as pd
import matplotlib.pyplot as plt

font = {
        'family' : 'normal',
        'size' : 6
}
plt.rc('font', **font)

Y = np.load("FirePkl/YQuintiles.npy")
Y = Y.astype(int)
X = np.load("FirePkl/X.npy")
X = X.astype(float)
XD = np.load("FirePkl/XDead.npy")
XD = XD.astype(float)
RF = np.loadtxt("TotalImportanceTopBinary.txt", dtype=str)
GElist = np.load("FirePkl/GElist.npy")
NonBinary = np.load("FirePkl/TotalImportanceTopNonBinary.npy")
NonBinaryInterface = np.load("FirePkl/RFTopInterface.npy")
Quint = np.load("TotalImportanceQuintilesauto.npy")
QuintNames = Quint[:,0]
QuintValues = Quint[:,1]
Quint1 = np.load("TotalImportanceautoforQuintile1.npy")
QuintNames1 = Quint1[:,0]
QuintValues1 = Quint1[:,1]
Quint2 = np.load("TotalImportanceautoforQuintile2.npy")
QuintNames2 = Quint2[:,0]
QuintValues2 = Quint2[:,1]
Quint3 = np.load("TotalImportanceautoforQuintile3.npy")
QuintNames3 = Quint3[:,0]
QuintValues3 = Quint3[:,1]
Quint4 = np.load("TotalImportanceautoforQuintile4.npy")
QuintNames4 = Quint4[:,0]
QuintValues4 = Quint4[:,1]
Quint5 = np.load("TotalImportanceautoforQuintile5.npy")
QuintNames5 = Quint5[:,0]
QuintValues5 = Quint5[:,1]


# NonBinary0 = NonBinary[:, 0]
# NonBinary1 = NonBinary[:,1]
# NonBinary1 = NonBinary1.astype(float)

# RF = np_f.replace(RF, "['", "")
# RF = np_f.replace(RF, '["', "")
# RF = np_f.replace(RF, "]", "")
# RF2 = [x[:-1] for x in RF[:,0]]
# RF3 = RF[:, 1]
# RF3 = RF3.astype(float)

# index0 = Y == -1
# index1 = Y > 0

top = 10

# Interface = np.zeros(len(RF)-3)
# k=0
# i=0
# while i < (len(RF2)-3):
#         if (RF2[k] == 'Treatment' or RF2[k] == 'Age' or RF2[k] == 'Stage'):
#                 k = k+1
#         else:
#                 temp = (np.where(RF2[k] == GElist)[0])
#                 Interface[i] = temp[0]
#                 k += 1
#                 i += 1
# Interface = Interface.astype(int)

# DX = X[index0, :]
# LX = X[index1, :]

# LAllAdvs = LX.sum(axis = 0)
# LGeneAdvs = LAllAdvs[Interface[:top]]
# LGeneAdvs = [x / LX.shape[0] for x in LGeneAdvs]

# DAllAdvs = X.sum(axis = 0)
# DGeneAdvs = DAllAdvs[Interface[:top]]
# DGeneAdvs = [x / DX.shape[0] for x in DGeneAdvs]

# GeneNames = GElist[Interface]
t= np.arange(top)

# plt.bar(t, height = DGeneAdvs)
# plt.ylabel('Mean Expression')
# plt.xticks(t, GeneNames)
# plt.title("Average Expression For the Top top Gene Predictors (Deceased Patients)")
# plt.savefig("Report/figure1.png", bbox_inches='tight')

# plt.clf()

# plt.bar(t, height = LGeneAdvs)
# plt.ylabel('Mean Expression')
# plt.xticks(t, GeneNames)
# plt.title("Average Expression For the Top top Gene Predictors (Alive Patients)")
# plt.savefig("Report/figure2.png", bbox_inches='tight')

# plt.clf()

# plt.bar(t, height = RF3[:top])
# plt.ylabel('Factor Weight')
# plt.xticks(t, GeneNames)
# plt.title("Average Weight of Each Gene on Prediction of Vital Status")
# plt.savefig("Report/figure3.png", bbox_inches='tight')

# plt.clf()


# plt.bar(t, height = NonBinary1[:10])
# plt.ylabel('Factor Weight')
# plt.xticks(t, NonBinary0[:10])
# plt.title("Average Weight of Each Gene on Prediction of Days to Death")
# plt.savefig("Report/figure4.png", bbox_inches='tight')

plt.bar(t, height = QuintValues[:top], align='center', width=0.3)
plt.ylabel("Factor Weight")
plt.xticks(t, QuintNames[:top])
plt.title("Weight of Top Ten Genes for Predicting Quintiles")
plt.savefig("TopTenGenesQuintiles.png", bbox_inches='tight')
plt.clf()

plt.bar(t, height = QuintValues1[:top], align='center', width=0.3)
plt.ylabel("Factor Weight")
plt.xticks(t, QuintNames1[:top])
plt.title("Weight of Top Ten Genes for Predicting Quintiles")
plt.savefig("Quintile1.png", bbox_inches='tight')
plt.clf()

plt.bar(t, height = QuintValues2[:top], align='center', width=0.3)
plt.ylabel("Factor Weight")
plt.xticks(t, QuintNames2[:top])
plt.title("Weight of Top Ten Genes for Predicting Quintiles")
plt.savefig("Quintile2.png", bbox_inches='tight')
plt.clf()

plt.bar(t, height = QuintValues3[:top], align='center', width=0.3)
plt.ylabel("Factor Weight")
plt.xticks(t, QuintNames3[:top])
plt.title("Weight of Top Ten Genes for Predicting Quintiles")
plt.savefig("Quintile3.png", bbox_inches='tight')
plt.clf()

plt.bar(t, height = QuintValues4[:top], align='center', width=0.3)
plt.ylabel("Factor Weight")
plt.xticks(t, QuintNames4[:top])
plt.title("Weight of Top Ten Genes for Predicting Quintiles")
plt.savefig("Quintile4.png", bbox_inches='tight')
plt.clf()

plt.bar(t, height = QuintValues5[:top], align='center', width=0.3)
plt.ylabel("Factor Weight")
plt.xticks(t, QuintNames5[:top])
plt.title("Weight of Top Ten Genes for Predicting Quintiles")
plt.savefig("Quintile5.png", bbox_inches='tight')
plt.clf()


#Quintile Top Overall features matching
QuintileInterface = np.load("FirePkl/RFTopInterfaceQuintiles.npy")
QuintileInterface = QuintileInterface.astype(int)

index = Y == 1
Y1 = Y[index]
X1 = XD[index, :]
avg = np.empty(top)
for i in range(top):
        avg[i] = np.sum(X1[:, QuintileInterface[i]])/X1.shape[1]

plt.bar(t, height = avg, align='center', width=0.3)
plt.ylabel("Factor Expression")
plt.xticks(t, GElist[QuintileInterface[:top]])
plt.title("Average Gene Expression of top Genes of Pateints in Quintile 1")
plt.savefig("PatientExpQuint1.png", bbox_inches='tight')
plt.clf()

index = Y == 2
Y2 = Y[index]
X2 = XD[index, :]
avg = np.empty(top)
for i in range(top):
        avg[i] = np.sum(X2[:, QuintileInterface[i]])/X2.shape[1]

plt.bar(t, height = avg, align='center', width=0.3)
plt.ylabel("Factor Expression")
plt.xticks(t, GElist[QuintileInterface[:top]])
plt.title("Average Gene Expression of top Genes of Pateints in Quintile 2")
plt.savefig("PatientExpQuint2.png", bbox_inches='tight')
plt.clf()

index = Y == 3
Y3 = Y[index]
X3 = XD[index, :]
avg = np.empty(top)
for i in range(top):
        avg[i] = np.sum(X3[:, QuintileInterface[i]])/X3.shape[1]

plt.bar(t, height = avg, align='center', width=0.3)
plt.ylabel("Factor Expression")
plt.xticks(t, GElist[QuintileInterface[:top]])
plt.title("Average Gene Expression of top Genes of Pateints in Quintile 3")
plt.savefig("PatientExpQuint3.png", bbox_inches='tight')
plt.clf()

index = Y == 4
Y4 = Y[index]
X4 = XD[index, :]
avg = np.empty(top)
for i in range(top):
        avg[i] = np.sum(X4[:, QuintileInterface[i]])/X4.shape[1]

plt.bar(t, height = avg, align='center', width=0.3)
plt.ylabel("Factor Expression")
plt.xticks(t, GElist[QuintileInterface[:top]])
plt.title("Average Gene Expression of top Genes of Pateints in Quintile 4")
plt.savefig("PatientExpQuint4.png", bbox_inches='tight')
plt.clf()

index = Y == 5
Y5 = Y[index]
X5 = XD[index, :]
avg = np.empty(top)
for i in range(top):
        avg[i] = np.sum(X5[:, QuintileInterface[i]])/X5.shape[1]

plt.bar(t, height = avg, align='center', width=0.3)
plt.ylabel("Factor Expression")
plt.xticks(t, GElist[QuintileInterface[:top]])
plt.title("Average Gene Expression of top Genes of Pateints in Quintile 5")
plt.savefig("PatientExpQuint5.png", bbox_inches='tight')
plt.clf()