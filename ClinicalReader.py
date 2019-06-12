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

# Days to death
DtD = Clin[22]
y = [x for x in DtD if str(x)!= 'nan']
y = y[1:]
y = np.asarray(y)
y = y.astype('int')
x = np.arange(0, len(y), 1)


# number survied and died
life = Clin[736]
life = life[1:]
count = np.in1d(life, 'alive').sum()
# splt.hist(y, bins=[0,1115, 2250, 3375, 4500])


# Patient ID dictionary
Cpat = Clin[17]
for i in range(len(Cpat)):
    Cpat[i] = Cpat[i].upper()

Etemp = Exp.columns
Epat = np.zeros(0, dtype = str)

for i in range(len(Etemp)):
    temp = Etemp[i]
    Epat = np.append(Epat, temp[:12])

BpatIndex = np.nonzero(np.in1d(Epat, Cpat))
Bpat = np.intersect1d(Epat, Cpat)

# num = np.arange(len(Bpat))
# PatDic = {}
# for num, Bpat in zip(num, Bpat):    #usless dicitonary
#     PatDic[num] = Bpat              #somehow also messes up Bpat


#List of all genes in same order as X
G = Exp["Hybridization REF"].values[1:]
G = G.astype(str)
normGenes = Norm["genes"].values
normGenes = normGenes.astype(str)

# print(G)
# print(normGenes)
# np.savetxt("FireBrowseGenes", G, fmt="%s")
# np.savetxt("MethylMixGenes", normGenes, fmt="%s")


# Normal Expression averages GEN
GEN = np.zeros(0, dtype = float)
# print(Norm)
# print(G)
inter = [x for x in G if x in Norm['genes'].values]
print(len(inter))
Norm['mean'] = Norm.mean(axis=1)
MeanList = Norm['mean'].values

# PatientXgene Matrix generation
newExp = Exp.to_numpy()
newExp = newExp[1:,1:]
newExp = np.transpose(newExp)
