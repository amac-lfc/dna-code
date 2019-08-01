import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys 

ClinD = pd.read_csv("../../FireBrowse/Clinical_Level/OV.clin.merged.txt", \
                    delimiter = "\t")
Clin = ClinD.to_numpy()
Exp = pd.read_csv("../../FireBrowse/OV.normalization_gene__data.data.txt", \
                    delimiter = "\t")
Norm = pd.read_csv("../../OVCancerDataSet/GEnormal_OV.csv", \
                    delimiter = ",")
Norm = Norm.rename(columns = {"Unnamed: 0": "genes"})

if len(sys.argv) == 1:
        Debug = True
else:
        Debug = sys.argv[1]

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

interface = np.zeros(len(Exp["Hybridization REF"].values), "i")

if Debug: print("interface length @ gen", interface.shape)

k=0
for i in range(len(Exp["Hybridization REF"].values)):
    interface[i] = np.where(Exp["Hybridization REF"].values[i] == Norm['genes'])[0]

if Debug: print("shape of interface after filling it", interface.shape)

# get only in common patients in EXP
fullNames = []
for i in range(len(Bpat)):
        fullNames.append(dicts.get(Bpat[i]))
fullNames = np.asarray(fullNames)
Exp = Exp[fullNames] 


if Debug: print("Exp after removal of no matching names", Exp.shape)

# GEN gen
GEN = np.zeros(0, dtype = float)
MeanList = Norm.mean(axis = 1)

if Debug: print(" shape of interface", interface.shape)
if Debug: print("len of mean list", len(MeanList))

MeanInter = np.zeros(len(interface), "d")

if Debug: print("shape of Mean Inter", MeanInter.shape)

for i in range(len(interface)):
        MeanInter[i] = MeanList[interface[i]]
GEN = MeanInter

if Debug: print("shape of GEN", GEN.shape)

#GE gen 
GE = np.zeros((len(interface),3))
GElist = np.empty(len(interface), dtype = "object")
for i in range(len(interface)):
        GE[i] = Norm.values[interface[i],1:] 
        GElist[i] = Norm.values[interface[i],0]
GE = np.transpose(GE)

if Debug: print("shape of GE", GE.shape)

# Y Gen
CIndex = []
for x in range(len(Exp.columns)):
    if x not in patIndexClin:
        CIndex.append(x)   

if Debug: print("number of items being deleted form both DtD and life"\
                ,len(CIndex))

DtD = np.delete(Clin[22], CIndex)
life = np.delete(Clin[736], CIndex)

if Debug: 
        print("# of items in life after", len(life))
        print("# of items in DtD after", len(DtD))

if Debug == True:
        print("number of patients in alive", life.shape)
        print("number of patients with days to death", DtD.shape)

tempL = []
for i in range(len(DtD)):
        if str(DtD[i]) == "nan" and str(life[i]) == "alive":
                 DtD[i] = -1
        elif str(DtD[i]) == "nan":
                DtD[i] = -2


temp = np.where(DtD == -2)
DtD = np.delete(DtD, temp)
print(DtD)
    

Exp = Exp.drop(Exp.columns[temp], axis = 1)

if Debug: print("patinets after merging DtD and life while removing NAN"\
                , DtD.shape)    
Y = DtD


#remove all patients not in Y

#convert newExp to X and make numpy matrix
newExp = Exp.to_numpy()

if Debug: print("newExp to numpy shape", newExp.shape)
newExp = newExp.astype(np.float)
newExp = np.transpose(newExp)

if Debug: print("new Exp after transposition", newExp.shape)
X = newExp

Age = np.delete(Clin[14][1:582], CIndex)
stages = np.delete(Clin[822][1:582], CIndex)
stageList = np.empty(len(stages), dtype = object)
for i in range(len(stages)):
    temp = stages[i]
    if str(temp) != 'nan':
        temp2 = temp[6:]
        stageList[i] = temp2
CancerStageDict = {
        "ia": 1,
        "ib": 2,
        "ic": 3,
        "iia": 4,
        "iib": 5,
        "iic": 6,
        "iiia": 7,
        "iiib": 8,
        "iiic": 9,
        "iv": 10,
        "None": -1
}
for i in range(len(stageList)):
        stageList[i] = CancerStageDict[str(stageList[i])]

ChemoList = np.delete(Clin[566][1:582], CIndex)
ChemoDict = {
        "chemotherapy": 1,
        "nan" : -1,
        "targeted molecular therapy" : 2,
        "hormone therapy" : 3,
        "immunotherapy" : 4
}
for i in range(len(ChemoList)):
        ChemoList[i] = ChemoDict[str(ChemoList[i])]

if Debug: print("Age and stage len", len(Age), len(stages))

X = np.column_stack((X, Age))
X = np.column_stack((X, stageList))
X = np.column_stack((X, ChemoList))

if Debug: print("shape of X for good measure", X.shape)



#DTD histogram
plt.clf()
Y2 = Y.astype(int)
print(np.amax(Y2))
plt.hist(Y2, bins=[-100, 0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
plt.savefig("Days to Death Histogram")

# printStatments
np.save("FirePkl/Y", Y, allow_pickle= True)
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