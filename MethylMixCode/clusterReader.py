import pandas as pd
import numpy as np

MET =  pd.read_csv("METcancer_OV.csv", delimiter = ",")
GE  = pd.read_csv("GEcancer_OV.csv", delimiter = ",")

METp = list(MET)
GEp = list(GE)

geneList = GE["Unnamed: 0"].values
inBoth = set(METp).intersection(GEp)

GE.loc[inBoth]

print(MET.shapes)
