import pandas as pd
import numpy as np

#reading the CVS Files
norm = pd.read_csv('computed_METnormal_OV.csv', delimiter="\t", \
                    names = ["gene", "Nmean", "NSD"])
cancer = pd.read_csv('computed_METcancer_OV.csv', delimiter="\t", \
                    names = ["gene", "Cmean", "CSD"])

mergedDF = pd.merge(norm, cancer, on='gene')

def condition(row):
    if row['Nmean'] == row['Cmean']:
        val= "NoChange"
    elif row['Nmean'] > row['Cmean']:
        val = "Down"
    else:
        val = "Up"
    return val

mergedDF['methelation'] = mergedDF.apply(condition, axis=1)

mergedDF.to_csv("mergedMethylStatus.csv")   
