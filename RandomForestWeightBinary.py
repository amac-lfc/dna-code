import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import random as rd

#importance fuction
def getImportances(classifier, X, features_list):
    importances = classifier.feature_importances_
    return(importances)
    

#number of genes used
top = 'auto'
#number of loops
N = 5

#file imports
Y = np.load("FirePkl/COADY.npy")
Y = Y.astype(int)
X = np.load("FirePkl/COADX.npy")
X = X.astype(float)
GElist = np.load("FirePkl/COADGenes.npy")
print(GElist)

alive = np.where(Y == 1)[0]
dead = np.where(Y == 0)[0]
print("Percent of patients alive TOTAL",len(alive)/len(Y))
print("Percent of paitents dead TOTAL", len(dead)/len(Y))

#Data Making
Data = np.column_stack((X, Y))

#Feature List
Features = np.copy(GElist)
def RFrun(Data, Y, i):

        #shuffle
        np.random.shuffle(Data)

        #train and test datasets
        test = Data[int(Data.shape[0] * .8):, :]
        train = Data[:int(Data.shape[0] * .8), :]
        

        where0 = np.where(train[:, -1]==0)[0]
        whereQ = np.where(train[:, -1]==1)[0]
        for k in range(int(len(whereQ)-len(where0))):
                temp = np.random.choice(where0)
                train = np.vstack((train, train[temp,:]))

        #train and test X and Y datasets
        Xtest = test[:, :-1]
        Ytest = test[:, -1]
        Xtrain = train[:, :-1]
        Ytrain = train[:, -1]
        alive = np.where(Ytest == 1)[0]
        dead = np.where(Ytest == 0)[0]
        print("Percent of patients alive TEST",len(alive)/len(Ytest))
        print("Percent of paitents dead TEST", len(dead)/len(Ytest))
        alive = np.where(Ytrain == 1)[0]
        dead = np.where(Ytrain == 0)[0]
        print("Percent of patients alive TRAIN",len(alive)/len(Ytrain))
        print("Percent of paitents dead TRAIN", len(dead)/len(Ytrain))

        #making the Classifiers
        clf_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_features=top, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
                oob_score=False, random_state=0, verbose=0, warm_start=False)

        #training
        clf_RF = clf_RF.fit(Xtrain, Ytrain)
        importanceArr = getImportances(clf_RF, Xtrain, Features)

        Ypredict = clf_RF.predict(Xtest)
        print(Ypredict)
        print(Ytest)
        print("Accuracy  is :", {accuracy_score(Ytest, Ypredict)}, i)
        return importanceArr

totalImportance = np.zeros(len(Features), 'd')
for i in range(N):
        totalImportance += RFrun(Data, Y, i)

totalImportance = totalImportance/N

toPrint = np.column_stack((Features[:len(Features)] ,(totalImportance[:])))

toPrint = toPrint[toPrint[:,1].argsort()[::-1]]

np.save("TotalImportanceCOAD", toPrint, allow_pickle=True)

f2 = open("TotalImportanceCOAD.txt", "w")
for i in range(toPrint.shape[0]):
    f2.write(str(toPrint[i,:]) + '\n')

# toPrint = toPrint[toPrint[:,0].argsort()[::-1]]

# # f3 = open("TotalImportanceTop"+str(top)+"ABC.txt", "w")
# # for i in range(top+3):
# #     f2.write(str(toPrint[i,:]) + '\n')
