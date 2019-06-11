import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ClinD = pd.read_csv("../../FireBrowse/Clinical_Level/OV.clin.merged.txt", \
                    delimiter = "\t")
Clin = ClinD.to_numpy()
Exp = pd.read_csv("../../FireBrowse/OV.normalization_gene__data.data.txt", \
                    delimiter = "\t")


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
Bpat = np.intersect1d(Epat, Cpat)

num = np.arange(len(Bpat))
PatDic = {}
for num, Bpat in zip(num, Bpat):
    PatDic[num] = Bpat




#Matrix generation
