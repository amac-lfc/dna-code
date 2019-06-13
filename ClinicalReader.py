import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ClinD = pd.read_csv("../../FireBrowse/Clinical_Level/OV.clin.merged.txt", \
                    delimiter = "\t")
Clin = ClinD.to_numpy()
Exp = pd.read_csv("../../FireBrowse/OV.normalization_gene__data.data.txt", \
                    delimiter = "\t")
Norm = pd.read_csv("../../OVCancerDataSet/GEnormal_OV.csv", \
                    delimiter = ",")
Norm = Norm.rename(columns = {"Unnamed: 0": "genes"})

#get only genes in both
G = Exp["Hybridization REF"].values[1:]
G = G.astype(str)
G = [x for x in G if x in Norm['genes'].values]

with open('In_Common_Genes', 'w') as f:
    for item in G:
        f.write("%s\n" % item)


# Patient ID dictionary
Cpat = Clin[17]
for i in range(len(Cpat)):
    Cpat[i] = Cpat[i].upper()

Etemp = Exp.columns
Epat = np.zeros(0, dtype = str)

for i in range(len(Etemp)):
    temp = Etemp[i]
    Epat = np.append(Epat, temp[:12])

B = np.intersect1d(Epat, Cpat, return_indices = True)
Bpat = B[0]
patIndex = B[1]

#List of all genes in same order as X
G = Exp["Hybridization REF"].values[1:]
G = G.astype(str)
normGenes = Norm["genes"].values
normGenes = normGenes.astype(str)

interface = np.zeros(len(Norm["genes"].values), "i")
k=0
for i in range(len(Norm["genes"].values)):
    temp = np.where(G == Norm['genes'].values[i])[0]
    if len(temp) != 0:
        interface[i] = temp[0]
    else:
        interface[i] = -1


np.savetxt("interfaceOFClinReader.txt", interface)
np.savetxt("FireBrowseGenes.txt", G, fmt="%s")
np.savetxt("MethylMixGenes.txt", Norm['genes'].values, fmt="%s")


# Normal Expression averages GEN
GEN = np.zeros(0, dtype = float)
Norm['mean'] = Norm.mean(axis=1)
MeanList = Norm['mean'].values
MeanInter = np.zeros(np.max(interface))
for i in range(len(interface)):
    # print(i, interface[i])
    if interface[i] != -1:
        MeanInter[interface[i]-1] = MeanList[i]

# PatientXgene Matrix generation
Bgene = np.intersect1d(normGenes, G)
Exp = Exp[~Exp["Hybridization REF"].isin(Bgene)]
OIndex = []
for x in range(len(Exp.columns)):
    if x not in patIndex:
        OIndex.append(x)
Exp = Exp.drop(Exp.columns[OIndex], axis =1)
newExp = Exp.to_numpy()
newExp = newExp[1:,1:]
newExp = np.transpose(newExp)

# Days to death
DtD = Clin[22]
y = [x for x in DtD if str(x)!= 'nan']
y = y[1:]
y = np.asarray(y)
y = y.astype('int')

# number survied and died
life = Clin[736]
life = life[1:]
count = np.in1d(life, 'alive').sum()
print(count+len(y))


#printStatments
# print(newExp)
# print(len(interface))
# print(len(MeanInter))
# print(newExp.shape)
# print(len(y))
