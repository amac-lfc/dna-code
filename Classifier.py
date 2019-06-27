import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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
top = 50  

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
    f1.write(GElist[SigIndex[i]] + "\t" + str(Y2CoreSig[SigIndex[i]])+'\n')

X2 = X[:, SigIndex[:top]]
X2 = np.column_stack((X2, X[:,-3:]))
Data = np.column_stack((X2, Y2))

#Feature List
Features = np.append(GElist[SigIndex[:top]], np.array(['Age', 'Stage', 'Treatment']))

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

#training
clf_RF = clf_RF.fit(Xtrain, Ytrain)
clf_tree = clf_tree.fit(Xtrain, Ytrain)
clf_svm = clf_svm.fit(Xtrain, Ytrain)

getImportances(clf_RF, Xtrain, Features, "RandomForestFeatures_"+str(top)+ '.txt')
getImportancesTrees(clf_tree, Xtrain, Features, "TreeFeatures_"+str(top)+ '.txt')
# getImportances(clf_svm, Xtrain, Features, "SVMFeatures_"+str(top)+ '.txt')

Ypredict = clf_RF.predict(Xtest)
print("Accuracy of Random Forest is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_tree.predict(Xtest)
print("Accuracy of Trees is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_svm.predict(Xtest)
print("Accuracy of SVC is :", {accuracy_score(Ytest, Ypredict)})

