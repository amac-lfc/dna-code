import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import random as rd

#importance fuction
def getImportances(classifier, X, features_list, targetFile):
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking and save in file
    print("Feature ranking:")
    t = open(targetFile, "w+")

    for f in range(X.shape[1]):
        t.write(f"{f + 1}. Feature {features_list[indices[f]]} ({importances[indices[f]]})"+"\n")

def getImportancesTrees(classifier, X, features_list, targetFile):
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking and save in file
    print("Feature ranking:")
    t = open(targetFile, "w+")

    for f in range(X.shape[1]):
        t.write(f"{f + 1}. Feature {features_list[indices[f]]} ({importances[indices[f]]})"+"\n")


#number of genes used
top = 17500
#number of loops
N = 50
#target quintile
Quintile = 5

#file imports
TopGenes = np.load("TotalImportanceQuintilesauto.npy")
TopGenesInterface = np.load("FirePkl/RFTopInterfaceQuintiles.npy")
TopGenesInterface = TopGenesInterface.astype(int)
Y = np.load("FirePkl/YQuintiles.npy")
Y = Y.astype(int)
index = Y!=Quintile
Y[index] = 0



X = np.load("FirePkl/XDead.npy")
X = X.astype(float)

X = X[:, TopGenesInterface[:top]]

GElist = np.load("FirePkl/GElist.npy")

Data = np.column_stack((X, Y))

Features = np.append(GElist[:], np.array(['Age', 'Stage', 'Treatment']))

#shuffle
np.random.shuffle(Data)

#data sets
test = Data[int(Data.shape[0] * .8):, :]
train = Data[:int(Data.shape[0] * .8), :]

#Data Duplication
where0 = np.where(train[:, -1]==0)[0]
whereQ = np.where(train[:, -1]==Quintile)[0]
for k in range(int(len(where0)-len(whereQ))):
    temp = np.random.choice(whereQ)
    train = np.vstack((train, train[temp,:]))

np.random.shuffle(train)

#making the Classifiers
clf_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

clf_tree = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

clf_svm = svm.SVC(gamma = 'scale')

clf_lr = LogisticRegression(solver= 'lbfgs', multi_class='auto', max_iter= 1000, tol= 1e-8 )

#training
Ytest = Data[int(Data.shape[0] * .8):, -1]
Xtest = Data[int(Data.shape[0] * .8):, :-1]
Xtrain = Data[:int(Data.shape[0] * .8), :-1]
Ytrain = Data[:int(Data.shape[0] * .8), -1]


param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000, 100000]
}


grid_search = GridSearchCV(estimator = clf_RF, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(Xtrain, Ytrain)
print(grid_search.best_params_)
clf_RF = grid_search.best_estimator_
#Instantiate the grid search model

clf_RF.fit(Xtrain, Ytrain)
clf_tree.fit(Xtrain, Ytrain)
clf_svm.fit(Xtrain, Ytrain)
clf_lr.fit(Xtrain, Ytrain)

getImportances(clf_RF, Xtrain, Features, "RandomForestFeatures_"+str(top)+ '.txt')
getImportancesTrees(clf_tree, Xtrain, Features, "TreeFeatures_"+str(top)+ '.txt')
# getImportances(clf_svm, Xtrain, Features, "SVMFeatures_"+str(top)+ '.txt')

Ypredict = clf_RF.predict(Xtest)
print("Accuracy of Random Forest is :", {accuracy_score(Ytest, Ypredict)})
print(Ytest)
print(Ypredict)

Ypredict = clf_tree.predict(Xtest)
print("Accuracy of Trees is :", {accuracy_score(Ytest, Ypredict)})
print(Ytest)
print(Ypredict)

Ypredict = clf_svm.predict(Xtest)
print("Accuracy of SVC is :", {accuracy_score(Ytest, Ypredict)})
print(Ytest)
print(Ypredict)

Ypredict = clf_lr.predict(Xtest)
print("Accuracy of Logistical Regression is :", {accuracy_score(Ytest, Ypredict)})
print(Ytest)
print(Ypredict)