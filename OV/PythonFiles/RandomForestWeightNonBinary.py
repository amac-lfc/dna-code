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
Y = np.load("FirePkl/Y.npy")
Y = Y.astype(int)

index = Y > 0
Y = Y[index]

X = np.load("FirePkl/X.npy")
X = X.astype(float)
X = X[index, :]


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
    clf_RF = tree.DecisionTreeRegressor(max_depth= None)

    #training
    clf_RF.fit(Xtrain, Ytrain)
    importanceArr = getImportances(clf_RF, Xtrain, Features, "RandomForestFeatures_NonBinary_"+str(top)+ '.txt')

    Ypredict = clf_RF.predict(Xtest)
    print(i)
    return importanceArr

totalImportance = np.zeros(len(Features), 'd')
for i in range(N):
        totalImportance += RFrun(Data, Y, i)

totalImportance = totalImportance/N

toPrint = np.column_stack((Features[:] ,(totalImportance[:])))

toPrint = toPrint[toPrint[:,1].argsort()[::-1]]

np.save("FirePkl/TotalImportanceTopNonBinary", toPrint, allow_pickle=True)

f2 = open("TotalImportanceTopNonBinary"+str(top)+".txt", "w")
for i in range(toPrint.shape[0]):
    f2.write(str(toPrint[i,:]) + '\n')
