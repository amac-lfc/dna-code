import pandas as pd
import numpy as np

MET =  pd.read_csv("METcancer_OV.csv", delimiter = ",")
GE  = pd.read_csv("GEcancer_OV.csv", delimiter = ",")

#changing name of gene collumn to gene
MET = MET.rename(columns = {"Unnamed: 0": "genes"})
GE = GE.rename(columns = {"Unnamed: 0": "genes"})

#removing all clusters from MET
MET = MET[~MET.genes.str.contains("Cluster")]

#removing the patients that arent in both
METp = list(MET.columns.values[1:])
GEp = list(GE.columns.values[1:])
inBoth = set(GEp).intersection(METp)
inBoth = list(inBoth)
inBoth = ["genes"] + inBoth
GE = GE[inBoth]
MET = MET[inBoth]


#removing genes that arent in both
mergedList = np.zeros(0, dtype = str)
GEgeneList = GE["genes"].values
METgeneList = MET["genes"].values

for i in range(len(METgeneList)):
    genome = METgeneList[i]
    temp = np.where(GEgeneList == genome)[0]
    mergedList = np.append(mergedList, GEgeneList[temp])

MET = MET[MET['genes'].isin(mergedList)]
GE = GE[GE['genes'].isin(mergedList)]

# generating the interface array
interface = np.zeros(len(METgeneList), dtype=int)

for i in range(len(MET['genes'].values)):
    temp = np.where(GE['genes'].values == MET['genes'].values[i])[0]
    interface[i] = (temp)[0]

MET.to_csv("METcancer_OV_processed.csv", index = False)
GE.to_csv("GEcancer_OV_processed.csv", index = False)
with open("Interface.txt", "w") as f:
    for item in interface:
        f.write("%s\n" % item)

MET.to_pickle("METcancer_OV_processed.pkl")
GE.to_pickle("GEcancer_OV_processed.pkl")
np.save("Interface.pkl", interface, allow_pickle = True)

MET = MET.drop(columns = ['genes'])
GE = GE.drop(columns = ['genes'])

np.save("Matrix_X.pkl", MET.values, allow_pickle = True)
np.save("Matrix_Y.pkl", GE.values, allow_pickle = True)
