import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys 

#import CSV files
ClinD = pd.read_csv("../../../FireBrowse/COAD/Clinical/COAD.clin.merged.txt", \
                    delimiter = "\t")
Clin = ClinD.to_numpy()
Exp = pd.read_csv("../../../FireBrowse/COAD/RNA/COAD.transcriptome__agilentg4502a_07_3__unc_edu__Level_3__unc_lowess_normalization_gene_level__data.data.txt", \
                    delimiter = "\t")
Methyl = pd.read_csv("../../../FireBrowse/COAD/Methyl/COAD.meth.by_mean.data.txt", \
                     delimiter="\t")
MethylD = Methyl.to_numpy()

#Debug if statment
Debug = False
# if len(sys.argv) == 1:
#         Debug = True
# else:
#         Debug = sys.argv[1]

Genes = Exp["Hybridization REF"].values[1:]

# days to death
DtD = Clin[21]

# Patient ID's in both Clin and Expression
Cpat = Clin[13]
for i in range(len(Cpat)):
    Cpat[i] = Cpat[i].upper()

if Debug: print("Clin patients", Cpat, Cpat.shape)

Etemp = Exp.columns
Epat = np.zeros(0, dtype = str)

# Mpat = Methyl.columns
# print(Mpat)
# for i in range(len(Mpat)):
#     Mpat[i] = Mpat[i].upper()
# print(Mpat)

dicts = {}
for i in range(len(Etemp)):
    temp = Etemp[i]
    dicts.update({temp[:12] : Etemp[i]})
    Epat = np.append(Epat, temp[:12])

if Debug: print("# of Exp patients", Epat.shape)

B = np.intersect1d(Epat, Cpat, return_indices = True)
Bpat = B[0]
patIndexExp = B[1] 
patIndexClin = B[2]


if Debug: print("List of patients in EXP and CLIN", Bpat, Bpat.shape)

print(patIndexClin)
print(patIndexExp)

ExpN = Exp.to_numpy()

Clin = Clin[:, patIndexClin]
ExpN = ExpN[:, patIndexExp]


print("after patient merge" ,ExpN.shape)

ExpN = np.transpose(ExpN)

VS = Clin[636,:]
ExpN = ExpN[:, 1:]
ExpN = ExpN.astype(float)

Y = np.zeros(len(VS))
for i in range(len(VS)):
    if str(VS[i]) == 'alive':
        Y[i] = 1
    else:
        Y[i] = 0

print('__________________________________________')
print(Genes)
print(Genes.shape)
print(Y)
print(Y.shape)
print(ExpN)
print(ExpN.shape)

index = np.isnan(ExpN)
ExpN[index] = 0

np.save('Pickles/COAD_DtD', DtD, allow_pickle=True)
np.save('Pickles/COADGenes', Genes, allow_pickle=True)
np.save("Pickles/COADY", Y, allow_pickle=True)
np.save("Pickles/COADX", ExpN, allow_pickle=True)

