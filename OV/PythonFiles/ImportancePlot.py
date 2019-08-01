import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
import random as rd
import sys
import matplotlib.pyplot as plt

#importance fuction
def getImportances(classifier, X, features_list, targetFile):
    importances = classifier.feature_importances_

    return(importances)
    

#number of genes used
top = 50
#number of loops
N = 2

if len(sys.argv)>1:
    top = int(sys.argv[1])
    print("top is {:d}".format(top))

if len(sys.argv)>2:
    N = int(sys.argv[2])
    print("N is {:d}".format(N))

#file imports
Y2CoreSig = np.load("FirePkl/Y2CoreSig.npy")
Y2 = np.load("FirePkl/Y2.npy")
Y = np.load("FirePkl/Y.npy")
X = np.load("FirePkl/X.npy")
X = X.astype(float)
GElist = np.load("FirePkl/GElist.npy")

Y = Y.astype(int)
# X = X.astype(float)

indices =  np.where(Y == -1)
X = np.delete(X, indices, axis=0)
Y = np.delete(Y, indices, axis=0)

# print(X)
# print(Y)


Y2Sorted = sorted(Y2CoreSig[:-4,1])
Y2Sorted = np.array(Y2Sorted)

SigIndex = np.zeros(len(Y2Sorted), dtype=int)
for i in range(len(Y2Sorted)):
    SigIndex[i] = np.where(Y2Sorted[i] == Y2CoreSig[:,1])[0][0]

f1 = open("SortedwName.txt", 'w+')
for i in range(len(SigIndex)):
    f1.write(GElist[SigIndex[i]] + "\t" + str(Y2CoreSig[SigIndex[i]])+'\n')

X2 = X[:, SigIndex[:top]]
X2 = np.column_stack((X2, X[:,-3:]))
Data = np.column_stack((X2, Y))

#Feature List
Features = np.append(GElist[SigIndex[:top]], np.array(['Age', 'Stage', 'Treatment']))

def RFrun(Data):

    #shuffle
    np.random.shuffle(Data)

    # print(Data.shape)

    DataTrain =  Data[:int(Data.shape[0] * .8), :]
    DataTest = Data[int(Data.shape[0] * .8):, :]
    DataTrain = DataTrain[DataTrain[:,-1].argsort()]
    DataTest = DataTest[DataTest[:,-1].argsort()]

    #train and test datasets
    Xtrain = DataTrain[:, :-1]
    Ytrain = DataTrain[:, -1]
    Xtest = DataTest[:, :-1]
    Ytest = DataTest[:, -1]


    #making the Classifiers
    clf_RF = tree.DecisionTreeRegressor(max_depth=100)

    rng = np.random.RandomState(1)
    clf_RF2 = AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=100),
                            n_estimators=1000, random_state=rng)

    #training
    clf_RF.fit(Xtrain, Ytrain)
    clf_RF2.fit(Xtrain, Ytrain)

    importanceArr = getImportances(clf_RF, Xtrain, Features, "RandomForestFeatures_"+str(top)+ '.txt')

    Ypredict = clf_RF.predict(Xtest)
    Ypredict2 = clf_RF2.predict(Xtest)
    # print(Ypredict)
    # print(Ytest)
    # print("Accuracy  is :", {accuracy_score(Ytest, Ypredict)})

    # cross_val_score(regressor, boston.data, boston.target, cv=10)


    X = range(len(Ytest))
    # Plot the results
    plt.figure()
    plt.scatter(X, Ytest, c="k", label="training samples")
    plt.plot(X, Ypredict, "g-*", label="n_estimators=1", linewidth=2)
    plt.plot(X, Ypredict2, "r", label="n_estimators=300", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show()

    return importanceArr

totalImportance = np.zeros(top+3, 'd')
for i in range(N):
    totalImportance += RFrun(Data)

totalImportance = totalImportance/N

toPrint = np.column_stack((Features[:] ,(totalImportance[:])))
toPrint = toPrint[toPrint[:,1].argsort()[::-1]]

f2 = open("TotalImportanceTop"+str(top)+".txt", "w")
for i in range(top+3):
    f2.write(str(toPrint[i,:]) + '\n')

toPrint = toPrint[toPrint[:,0].argsort()[::-1]]

f3 = open("TotalImportanceTop"+str(top)+"ABC.txt", "w")
for i in range(top+3):
    f2.write(str(toPrint[i,:]) + '\n')
