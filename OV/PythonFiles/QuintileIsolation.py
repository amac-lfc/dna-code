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
#target quintile
Quintile = 1

#file imports
Y = np.load("FirePkl/YQuintiles.npy")
Y = Y.astype(int)


index = Y!=Quintile
Y[index] = 0


X = np.load("FirePkl/XDead.npy")
X = X.astype(float)


GElist = np.load("FirePkl/GElist.npy")

where0 = np.where(Y==0)[0]
whereQ = np.where(Y==Quintile)[0]

Data = np.column_stack((X, Y))

#Feature List
Features = np.append(GElist[:], np.array(['Age', 'Stage', 'Treatment']))
def RFrun(Data, Y, i):

    Data2 = np.copy(Data)

    for k in range(len(where0)-len(whereQ)):
        temp = np.random.choice(whereQ)
        Data2 = np.vstack((Data2, Data2[temp,:]))


    #shuffle
    np.random.shuffle(Data2)

    #train and test datasets
    Xtrain = Data2[:int(Data2.shape[0] * .8), :-1]
    Ytrain = Data2[:int(Data2.shape[0] * .8), -1]
    Xtest = Data2[int(Data2.shape[0] * .8):, :-1]
    Ytest = Data2[int(Data2.shape[0] * .8):, -1]

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

np.save("TotalImportance" + str(top)+ "forQuintile" + str(Quintile), toPrint, allow_pickle=True)

f2 = open("TotalImportance" + str(top)+ "forQuintile" + str(Quintile), "w")
for i in range(toPrint.shape[0]):
    f2.write(str(toPrint[i,:]) + '\n')