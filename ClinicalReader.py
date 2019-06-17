import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ClinD = pd.read_csv("../../FireBrowse/Clinical_Level/OV.clin.merged.txt", \
                    delimiter = "\t")
Clin = ClinD.to_numpy()
Exp = pd.read_csv("../../FireBrowse/OV.normalization_gene__data.data.txt", \
                    delimiter = "\t")
Norm = pd.read_csv("../../OVCancerDataSet/GEnormal_OV.csv", \
                    delimiter = ",")
Norm = Norm.rename(columns = {"Unnamed: 0": "genes"})

Debug = True

temp = np.where(pd.isnull(Clin[736])) 
Clin = np.delete(Clin, [temp], axis =1)


if Debug == True:
        print("CLinD shape 18", ClinD.shape)
        print("exp shape", Exp.shape)
        print("Norm shape",Norm.shape)

#get only genes in both Expresion and Norm
G = Exp["Hybridization REF"].values[1:]
G = G.astype(str)
G = [x for x in G if x in Norm['genes'].values]
if Debug: print("shape of G line 16", len(G))

# Patient ID dictionary
Cpat = Clin[17]
for i in range(len(Cpat)):
    Cpat[i] = Cpat[i].upper()

if Debug: print("Clin patients", Cpat.shape)

Etemp = Exp.columns
Epat = np.zeros(0, dtype = str)

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

if Debug: print("List of both patients", Bpat.shape)

# PatientXgene Matrix generation

Exp = Exp[Exp["Hybridization REF"].isin(G)]

if Debug: print("Exp after removal of non simialr genes", Exp.shape)

#Interface Gen
normGenes = Norm["genes"].values
normGenes = normGenes.astype(str)

if Debug: print("norm genes length", normGenes.shape )

interface = np.zeros(len(Norm["genes"].values), "i")
k=0
for i in range(len(Norm["genes"].values)):
    temp = np.where(Exp["Hybridization REF"] == Norm['genes'].values[i])[0]
    if len(temp) != 0:
        interface[i] = temp[0]
    else: 
        interface[i] = -1

fullNames = []
for i in range(len(Bpat)):
        fullNames.append(dicts.get(Bpat[i]))
fullNames = np.asarray(fullNames)
Exptemp = Exp[fullNames] 
#not touched again untill the end when converted to Numpy matrix

if Debug: print("shape of interface", interface.shape)

if Debug: print("Exp after removal of no matching names", Exptemp.shape)

# GEN gen
GEN = np.zeros(0, dtype = float)
MeanList = Norm.mean(axis = 1)

MeanInter = np.zeros(np.max(interface))
for i in range(len(interface)):
    if interface[i] != -1:
        MeanInter[interface[i]-1] = MeanList[i]
GEN = MeanInter[np.nonzero(MeanInter)]

#GE gen 
GE = np.zeros((len(GEN),3))
GElist = np.zeros( len(GEN), dtype = str)
for i in range(len(interface)):
    if interface[i] != -1:
        GE[interface[i]-1] = Norm.values[i,1:] 
        GElist[interface[i]-1] = Norm.values[i,0]
GE = np.transpose(GE)

# Y Gen
CIndex = []
for x in range(len(Exp.columns)):
    if x not in patIndexClin:
        CIndex.append(x)   
if Debug: print("number of items being deleted form both DtD and life"\
                ,len(CIndex))
DtD = np.delete(Clin[22], CIndex)
life = np.delete(Clin[736], CIndex)

if Debug == True:
        print("number of patients in alive", life.shape)
        print("number of patients with days to death", DtD.shape)

tempL = []
for i in range(len(DtD)):
        if str(DtD[i]) == "nan" and str(life[i]) == "alive":
                DtD[i] = -1
        elif str(DtD[i]) == "nan":
                tempL.append(i)

DtD = np.delete(DtD, tempL)

if Debug: print("patinets after merging DtD and life while removing NAN"\
                , DtD.shape)    
Y = DtD


#convert newExp to X and make numpy matrix
newExp = Exptemp.to_numpy()

if Debug: print("newExp to numpy shape", newExp.shape)
newExp = newExp.astype(np.float)
newExp = np.transpose(newExp)

if Debug: print("new Exp after transposition", newExp.shape)
X = newExp

if Debug: print("shape of X for good measure", X.shape)

# printStatments
np.save("FirePkl/y", Y, allow_pickle= True)
np.save("FirePkl/X", X, allow_pickle= True)
np.save("FirePkl/GEN", GEN, allow_pickle= True)
np.save("FirePkl/GE", GE, allow_pickle= True)
np.save("FirePkl/GElist", GElist, allow_pickle= True)

#print all the shapes
print("print of all the shapes")
print("shape of GElist =" , GElist.shape)
print("shape of GE =" ,GE.shape)
print("shape of GEN =" ,GEN.shape)
print("shape of X =" ,X.shape)
print("shape of Y =" , Y.shape)