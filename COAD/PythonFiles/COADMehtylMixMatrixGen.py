import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys 

#import CSV files
Exp = pd.read_csv("../../../epigenome_regulation/COAD/COAD.uncv2.mRNAseq_RSEM_Z_Score.txt", \
                    delimiter = "\t")
Methyl = pd.read_csv("../../../epigenome_regulation/COAD/COAD.methylation__humanmethylation27__jhu_usc_edu__Level_3__within_bioassay_data_set_function__data.data.txt", \
                     delimiter="\t")
MethylD = Methyl.to_numpy()

#Debug if statment
Debug = False
# if len(sys.argv) == 1:
#         Debug = True
# else:
#         Debug = sys.argv[1]

Genes = Exp["Hybridization REF"].values[1:]

# #isolate predictors
# Age = Clin[11]
# stages = Clin[767]

# if Debug: print("Ages:", Age)
# if Debug: print("Stages :", stages)
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

if Debug: print("List of patients in EXP and CLIN", Bpat, Bpat.shape)

print(patIndexClin)
print(patIndexExp)

ExpN = Exp.to_numpy()

ExpN = ExpN[:, patIndexExp]


print("after patient merge" ,ExpN.shape)

ExpN = np.transpose(ExpN)

ExpN = ExpN[:, 1:]
ExpN = ExpN.astype(float)

print('__________________________________________')
print(Genes)
print(Genes.shape)
print(Y)
print(Y.shape)
print(ExpN)
print(ExpN.shape)

index = np.isnan(ExpN)
ExpN[index] = 0

np.save('../FirePkl/MMCOADGenes', Genes, allow_pickle=True)
np.save("../FirePkl/MMCOADY", Y, allow_pickle=True)
np.save("../FirePkl/MMCOADX", ExpN, allow_pickle=True)

# # A = np.intersect1d(Bpat, Mpat, return_indices=True)

# # # #get only genes in both Expresion and Methyl
# # G = Exp["Hybridization REF"].values[5:]
# # sep = "|"
# # for i in range(len(G)):
# #     G[i] = G[i].split(sep, 1)[0]
# # T = Methyl['Hybridization REF'].values
# # # print(G, T)
# # G = G.astype(str)
# # G = [x for x in G if x in Methyl['Hybridization REF'].values]
# # if Debug: print("shape of G line 16", len(G))
