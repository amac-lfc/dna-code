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
top = 'auto'
#number of loops
N = 50

#file imports
# Y2CoreSig = np.load("FirePkl/Y2CoreSig.npy")

Y = np.load("FirePkl/Y.npy")
Y = Y.astype(int)
index = Y == -1
Y[index] = 0
index = Y > 0
Y[index] = 1

X = np.load("FirePkl/X.npy")
X = X.astype(float)

GElist = np.load("FirePkl/GElist.npy")

Data = np.column_stack((X, Y))

#Feature List
Features = np.append(GElist[:], np.array(['Age', 'Stage', 'Treatment']))
def RFrun(Data, Y, i):

    #shuffle
    np.random.shuffle(Data)

    #train and test datasets
    Xtrain = Data[:int(Data.shape[0] * .8), :-1]
    Ytrain = Data[:int(Data.shape[0] * .8), -1]
    Xtest = Data[int(Data.shape[0] * .8):, :-1]
    Ytest = Data[int(Data.shape[0] * .8):, -1]

    #making the Classifiers
    clf_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_features=top, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

    #training
    clf_RF = clf_RF.fit(Xtrain, Ytrain)
    importanceArr = getImportances(clf_RF, Xtrain, Features, "RandomForestFeatures_"+str(top)+ '.txt')

    Ypredict = clf_RF.predict(Xtest)
    print("Accuracy  is :", {accuracy_score(Ytest, Ypredict)}, i)
    return importanceArr

totalImportance = np.zeros(len(Features), 'd')
for i in range(N):
        totalImportance += RFrun(Data, Y, i)

totalImportance = totalImportance/N

toPrint = np.column_stack((Features[:] ,(totalImportance[:])))

toPrint = toPrint[toPrint[:,1].argsort()[::-1]]

np.save("TotalImportanceTop" + str(top)+ ' .txt', toPrint, allow_pickle=True)

f2 = open("TotalImportanceTop"+str(top)+".txt", "w")
for i in range(toPrint.shape[0])
    f2.write(str(toPrint[i,:]) + '\n')

# toPrint = toPrint[toPrint[:,0].argsort()[::-1]]

# # f3 = open("TotalImportanceTop"+str(top)+"ABC.txt", "w")
# # for i in range(top+3):
# #     f2.write(str(toPrint[i,:]) + '\n')
