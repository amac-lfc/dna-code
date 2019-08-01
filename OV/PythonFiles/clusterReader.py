import pandas as pd
import numpy as np

MET =  pd.read_csv("METcancer_OV.csv", delimiter = ",")
GE  = pd.read_csv("GEcancer_OV.csv", delimiter = ",")

METp = list(MET.columns.values)
GEp = list(GE.columns.values)

geneList = GE["Unnamed: 0"].values
inBoth = set(GEp).intersection(METp)

inBoth = list(inBoth)
GE = GE[inBoth]

MET = MET[inBoth]


MET = MET.reindex(sorted(MET.columns), axis=1)
GE = GE.reindex(sorted(GE.columns), axis=1)

MET.to_csv("MatrixX.csv")
GE.to_csv("MatrixY.csv")
# print(MET)
# print(GE)
