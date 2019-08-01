import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys 

#import CSV files
ClinD = pd.read_csv("../../FireBrowse/COAD/Clinical/COAD.clin.merged.txt", \
                    delimiter = "\t")
Clin = ClinD.to_numpy()
Exp = pd.read_csv("../../FireBrowse/COAD/RNA/COAD.rnaseqv2__illuminaga_rnaseqv2__unc_edu__Level_3__RSEM_genes__data.data.txt", \
                    delimiter = "\t")
Methyl = pd.read_csv("../../FireBrowse/COAD/Methyl/COAD-FFPE.methylation__humanmethylation450__jhu_usc_edu__Level_3__within_bioassay_data_set_function__data.data.txt", \
                     delimiter="\t")
Norm = pd.read_csv("../../OVCancerDataSet/GEnormal_OV.csv", \
                    delimiter = ",")
Norm = Norm.rename(columns = {"Unnamed: 0": "genes"})

print(ClinD)
print(ClinD.shape)
print(Exp)
print(Exp.shape)
print(Methyl)
print(Methyl.shape)
print(Norm)
print(Norm.shape)



# #Debug if statment
# if len(sys.argv) == 1:
#         Debug = True
# else:
#         Debug = sys.argv[1]


# #isolate predictors
# Age = Clin[14][1:582]
# stages = Clin[822][1:582]


# # Patient ID's in both Clin and Expression
# Cpat = Clin[17]
# for i in range(len(Cpat)):
#     Cpat[i] = Cpat[i].upper()

# if Debug: print("Clin patients", Cpat.shape)

# Etemp = Exp.columns
# Epat = np.zeros(0, dtype = str)

# dicts = {}
# for i in range(len(Etemp)):
#     temp = Etemp[i]
#     dicts.update({temp[:12] : Etemp[i]})
#     Epat = np.append(Epat, temp[:12])

# if Debug: print("# of Exp patients", Epat.shape)

# B = np.intersect1d(Epat, Cpat, return_indices = True)
# Bpat = B[0]
# patIndexExp = B[1] 
# patIndexClin = B[2]

# if Debug: print("List of both patients", Bpat.shape)


# #get only genes in both Expresion and Norm
# G = Exp["Hybridization REF"].values[1:]
# G = G.astype(str)
# G = [x for x in G if x in Norm['genes'].values]
# if Debug: print("shape of G line 16", len(G))