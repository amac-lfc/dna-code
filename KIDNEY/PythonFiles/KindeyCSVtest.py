import numpy as np
import pandas as pd

Clin = pd.read_csv("KIPAN.clin.merged.txt", delimiter='\t', low_memory=False)
Methyl = pd.read_csv("KIPAN.methylation__humanmethylation450__jhu_usc_edu__Level_3__within_bioassay_data_set_function__data.data.txt", delimiter='\t')
Exp = pd.read_csv("KIPAN.rnaseq__illuminahiseq_rnaseq__unc_edu__Level_3__gene_expression__data.data.txt", delimiter='\t', low_memory=False)

print(Clin)
print(Clin.shape)
print(Methyl)
print(Methyl.shape)
print(Exp)
print(Exp.shape)