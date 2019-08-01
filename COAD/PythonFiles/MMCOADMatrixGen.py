import pandas as pd
import numpy as np
import scipy.stats as sy

# load the CSV and get all gene names
Exp = pd.read_csv("../../epigenome_regulation/COAD/COAD.uncv2.mRNAseq_RSEM_Z_Score.txt", \
                    delimiter = "\t")
Methyl = pd.read_csv('../../epigenome_regulation/COAD/COAD.methylation__humanmethylation27__jhu_usc_edu__Level_3__within_bioassay_data_set_function__data.data.txt', \
        delimiter='\t')

ExpGenes = Exp['gene'].values[1:]
ExpNumpy = Exp.to_numpy()[1:,1:]
ExpNumpy = np.transpose(ExpNumpy)
ExpPatients = Exp.columns[1:]
print(ExpNumpy)
ExpNumpy = ExpNumpy.astype(float)
ExpMean = np.mean(ExpNumpy, axis = 0)

np.save('FirePkl/MMCOADExpGenes', ExpGenes, allow_pickle=True)
np.save('FirePkl/MMCOADExpPatients', ExpPatients, allow_pickle=True)
np.save('FirePkl/MMCOADExpValues', ExpNumpy, allow_pickle=True)
np.save('FirePkl/MMCOADExpMean', ExpMean, allow_pickle=True)

MethylPatients = Methyl.columns[1::4]
MethylNumpy = Methyl.to_numpy()
MethylNumpy = MethylNumpy[:,1:]
MethylGenes = MethylNumpy[1:,1]

MethylValues = np.zeros((len(MethylPatients), len(MethylGenes)))

for j in range(len(MethylGenes)):
        for i in range(len(MethylPatients)):
               MethylValues[i,j] = float(MethylNumpy[j+1,4*i])

MethylMean = np.mean(MethylValues, axis = 0)

MethylPatientsCutOff = np.empty(len(MethylPatients), dtype=object)
for i in range(len(MethylPatients)):
    MethylPatientsCutOff[i] = str(MethylPatients[i][:15])

print(MethylPatientsCutOff)
print(ExpPatients)
c = 0
for i in range(len(MethylPatientsCutOff)):
    for j in range(len(ExpPatients)):
        if str(MethylPatientsCutOff[i]) == str(ExpPatients[j]):
            c +=1
print(c)

np.save('FirePkl/MMCOADMethylGenes', MethylGenes, allow_pickle=True)
np.save('FirePkl/MMCOADMethylPatients', MethylPatients, allow_pickle=True)
np.save('FirePkl/MMCOADMethylValues', MethylValues, allow_pickle=True)
np.save('FirePkl/MMCOADMethylMean', MethylMean, allow_pickle=True)