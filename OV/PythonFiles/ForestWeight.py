import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import random as rd

#importance fuction
def getImportances(classifier, X, features_list, targetFile):
    importances = classifier.feature_importances_
    return(importances)
    

#number of genes used
top = 10

#file imports
Y2CoreSig = np.load("FirePkl/Y2CoreSig.npy")
Y2 = np.load("FirePkl/Y2.npy")
Y = np.load("FirePkl/Y.npy")
X = np.load("FirePkl/X.npy")
X = X.astype(float)
GElist = np.load("FirePkl/GElist.npy")

Y2Sorted = sorted(Y2CoreSig[:-4,1])
Y2Sorted = np.array(Y2Sorted)

SigIndex = np.zeros(len(Y2Sorted), dtype=int)
for i in range(len(Y2Sorted)):
    SigIndex[i] = np.where(Y2Sorted[i] == Y2CoreSig[:,1])[0][0]

f1 = open("SortedwName.txt", 'w+')
for i in range(len(SigIndex)):
    f1.write(GElist[SigIndex[i]] + "\t" + str(Y2CoreSig[SigIndex[i]]) + '\n')

X2 = X[:, SigIndex[:top]]
X2 = np.column_stack((X2, X[:,-3:]))
Data = np.column_stack((X2, Y2))

#Feature List
Features = np.append(GElist[SigIndex[:top]], np.array(['Age', 'Stage', 'Treatment']))

for i in range(len(Features)):
        if str(Features[i]) == "Age":
                print("At index "+ str(i) + 'at creation')

def RFrun(Data, Y2):

    Y2  = Y2.astype(int)
    MaxY2 = np.max(Y2)
    Maxes = []
    mask = []
    for i in range(MaxY2+1):
            mask.append(np.where(Y2==i)[0])
            Maxes.append(len(mask[i]))
            # print("Y == {:d} has {:d} values".format(i,Maxes[i]))

    Max = np.max(Maxes)
    # print(Max)
    for i in range(MaxY2+1):
            for j in range(Max-Maxes[i]):
                    k = rd.choice(mask[i])
                    Data = np.concatenate((Data, Data[k,:][np.newaxis,:]), axis=0)

    # Maxes = []
    # mask = []
    # for i in range(MaxY2+1):
    #         mask.append(np.where(Data[:,-1]==i)[0])
    #         Maxes.append(len(mask[i]))
    #         print("Y == {:d} has {:d} values".format(i,Maxes[i]))


    #shuffle
    np.random.shuffle(Data)

    #train and test datasets
    Xtrain = Data[:int(Data.shape[0] * .8), :-1]
    Ytrain = Data[:int(Data.shape[0] * .8), -1]
    Xtest = Data[int(Data.shape[0] * .8):, :-1]
    Ytest = Data[int(Data.shape[0] * .8):, -1]

    #making the Classifiers
    clf_tree = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

    #training
    clf_tree = clf_tree.fit(Xtrain, Ytrain)

    importanceArr = getImportances(clf_tree, Xtrain, Features, "RandomForestFeatures_"+str(top)+ '.txt')

    Ypredict = clf_tree.predict(Xtest)
    print("Accuracy  is :", {accuracy_score(Ytest, Ypredict)})

    return importanceArr

totalImportance = np.zeros(top+3, 'd')
for i in range(500):
    totalImportance += RFrun(Data, Y2)

totalImportance = totalImportance/100

toPrint = np.column_stack((Features[:] ,(totalImportance[:])))
toPrint = toPrint[toPrint[:,1].argsort()[::-1]]

for i in range(len(Features)):
        if str(Features[i]) == "Age":
                print("At index "+ str(i) + " Before print")

f2 = open("TotalImportanceTop"+str(top)+".txt", "w+")
for i in range(top+3):
    f2.write(str(toPrint[i,:]) + '\n')

toPrint = toPrint[toPrint[:,0].argsort()[::-1]]

for i in range(len(Features)):
        if str(Features[i]) == "Age":
                print("At index "+ str(i) + " Before 2nd print")

f3 = open("TotalImportanceTop"+str(top)+"ABC.txt", "w+")
for i in range(top+3):
    f2.write(str(toPrint[i,:]) + '\n')
