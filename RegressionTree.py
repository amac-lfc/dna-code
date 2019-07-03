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
top = 10

#file imports
RF = np.load("FirePkl/RFTopInterface.npy")
RF = RF.astype(int)
Y = np.load("FirePkl/Y.npy")
X = np.load("FirePkl/X.npy")
X = X.astype(float)
GElist = np.load("FirePkl/GElist.npy")
Y = Y.astype(int)
X = X.astype(float)

indices =  np.where(Y == -1)
X = np.delete(X, indices, axis=0)
Y = np.delete(Y, indices, axis=0)

X2 = X[:, RF[:top]]
X2 = np.column_stack((X2, X[:,-3:]))
Data = np.column_stack((X2, Y))

#Feature List
Features = np.append(GElist[RF[:top]], np.array(['Age', 'Stage', 'Treatment']))

#shuffle
np.random.shuffle(Data)

# print(Data.shape)

DataTrain =  Data[:int(Data.shape[0] * .8), :]
DataTest = Data[int(Data.shape[0] * .8):, :]
# DataTrain = DataTrain[DataTrain[:,-1].argsort()]
DataTest = DataTest[DataTest[:,-1].argsort()]

#train and test datasets
Xtrain = DataTrain[:, :-1]
Ytrain = DataTrain[:, -1]
Xtest = DataTest[:, :-1]
Ytest = DataTest[:, -1]


#making the Classifiers
clf_RF = tree.DecisionTreeRegressor(max_depth=None)

rng = np.random.RandomState(1)
clf_RF2 = AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=None),
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
plt.plot(X, Ypredict, "g-*", label="Random Forest Regression", linewidth=2)
plt.plot(X, Ypredict2, "r", label="Boosted Random Forest Regression", linewidth=2)
plt.xlabel("Patients")
plt.ylabel("Days to Death")
# plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
