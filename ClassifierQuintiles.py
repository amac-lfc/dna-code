import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
top = 100

#file imports
Y = np.load("FirePkl/Y.npy")
Y = Y.astype(int)
X = np.load("FirePkl/X.npy")
X = X.astype(float)
GElist = np.load("FirePkl/GElist.npy")
RFTopInterface = np.load("FirePkl/RFTopInterfaceQuintiles.npy")
RFTopInterface = RFTopInterface.astype(int)

X2 = X[:, RFTopInterface[:top]]
X2 = np.column_stack((X2, X[:,-3:]))
Data = np.column_stack((X2, Y))


#Feature List
Features = np.append(GElist[RFTopInterface[:top]], np.array(['Age', 'Stage', 'Treatment']))


#shuffle
np.random.shuffle(Data)

#train and test datasets
Xtrain = Data[:int(Data.shape[0] * .8), :-1]
Ytrain = Data[:int(Data.shape[0] * .8), -1]
Xtest = Data[int(Data.shape[0] * .8):, :-1]
Ytest = Data[int(Data.shape[0] * .8):, -1]

#making the Classifiers
clf_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
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
clf_RF = clf_RF.fit(Xtrain, Ytrain)
clf_tree = clf_tree.fit(Xtrain, Ytrain)
clf_svm = clf_svm.fit(Xtrain, Ytrain)
clf_lr = clf_lr.fit(Xtrain, Ytrain)

getImportances(clf_RF, Xtrain, Features, "RandomForestFeatures_"+str(top)+ '.txt')
getImportancesTrees(clf_tree, Xtrain, Features, "TreeFeatures_"+str(top)+ '.txt')
# getImportances(clf_svm, Xtrain, Features, "SVMFeatures_"+str(top)+ '.txt')

Ypredict = clf_RF.predict(Xtest)
print("Accuracy of Random Forest is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_tree.predict(Xtest)
print("Accuracy of Trees is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_svm.predict(Xtest)
print("Accuracy of SVC is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_lr.predict(Xtest)
print("Accuracy of Logistical Regression is :", {accuracy_score(Ytest, Ypredict)})
