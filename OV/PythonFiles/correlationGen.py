import pandas as pd
import numpy as np
import scipy.stats as sy

# load the CSV and get all gene names
Exp = pd.read_csv('../../FireBrowse/OV/OV.transcriptome__ht_hg_u133a__broad_mit_edu__Level_3__gene_rma__data.data.txt', 
        delimiter='\t')
Methyl = pd.read_csv('../../FireBrowse/OV/OV.methylation__humanmethylation27__jhu_usc_edu__Level_3__within_bioassay_data_set_function__data.data.txt', \
        delimiter='\t')

ExpGenes = Exp['Hybridization REF'].values[1:]
ExpNumpy = Exp.to_numpy()[1:,1:]
ExpNumpy = np.transpose(ExpNumpy)
ExpPatients = Exp.columns[1:]
print(ExpNumpy)
ExpNumpy = ExpNumpy.astype(float)
ExpMean = np.mean(ExpNumpy, axis = 0)

np.save('FirePkl/OVExpGenes', ExpGenes, allow_pickle=True)
np.save('FirePkl/OVExpPatients', ExpPatients, allow_pickle=True)
np.save('FirePkl/OVExpValues', ExpNumpy, allow_pickle=True)
np.save('FirePkl/OVExpMean', ExpMean, allow_pickle=True)

MethylPatients = Methyl.columns[1::4]
MethylNumpy = Methyl.to_numpy()
MethylNumpy = MethylNumpy[:,1:]
MethylGenes = MethylNumpy[1:,1]

MethylValues = np.zeros((len(MethylPatients), len(MethylGenes)))

for j in range(len(MethylGenes)):
        for i in range(len(MethylPatients)):
               MethylValues[i,j] = float(MethylNumpy[j+1,4*i])

MethylMean = np.mean(MethylValues, axis = 0)

np.save('FirePkl/OVMethylGenes', MethylGenes, allow_pickle=True)
np.save('FirePkl/OVMethylPatients', MethylPatients, allow_pickle=True)
np.save('FirePkl/OVMethylValues', MethylValues, allow_pickle=True)
np.save('FirePkl/OvMethylMean', MethylMean, allow_pickle=True)