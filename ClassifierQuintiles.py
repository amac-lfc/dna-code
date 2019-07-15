import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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
top = 500


#file imports
Y = np.load("FirePkl/YQuintiles.npy")
Y = Y.astype(int)


X = np.load("FirePkl/XDead.npy")
X = X.astype(float)
print(X.shape)  

GElist = np.load("FirePkl/GElist.npy")

Data = np.column_stack((X, Y))


#Feature List
Features = np.append(GElist[:], np.array(['Age', 'Stage', 'Treatment']))
print(Features.shape)

#shuffle
np.random.shuffle(Data)

#Y into arrays
newY = np.zeros((len(Y), 5), dtype=int)
for i in range(len(Y)):
    if Y[i] == 1:
        newY[i, 0] = 1
    elif Y[i] == 2:
        newY[i, 1] = 1
    elif Y[i] == 3:
        newY[i, 2] = 1
    elif Y[i] == 4:
        newY[i, 3] = 1
    else:
        newY[i, 4] = 1

#train and test datasets
Xtrain = Data[:int(Data.shape[0] * .8), :-1]
print(Xtrain.shape)
Ytrain = newY[:int(newY.shape[0]* .8), :]
print(Ytrain.shape)
Xtest = Data[int(Data.shape[0] * .8):, :-1]
print(Xtest.shape)
Ytest = newY[int(newY.shape[0]* .8):, :]
print(Ytest.shape)

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

clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-10, hidden_layer_sizes=(1700, 2    ), random_state=1)

#training
# clf_RF.fit(Xtrain, Ytrain)
# clf_tree.fit(Xtrain, Ytrain)
# clf_svm.fit(Xtrain, Ytrain)
# clf_lr.fit(Xtrain, Ytrain)
clf_MLP.fit(Xtrain, Ytrain)

# getImportances(clf_RF, Xtrain, Features, "RandomForestFeatures_"+str(top)+ '.txt')
# getImportancesTrees(clf_tree, Xtrain, Features, "TreeFeatures_"+str(top)+ '.txt')
# # getImportances(clf_svm, Xtrain, Features, "SVMFeatures_"+str(top)+ '.txt')

# Ypredict = clf_RF.predict(Xtest)
# print("Accuracy of Random Forest is :", {accuracy_score(Ytest, Ypredict)})

# Ypredict = clf_tree.predict(Xtest)
# print("Accuracy of Trees is :", {accuracy_score(Ytest, Ypredict)})

# Ypredict = clf_svm.predict(Xtest)
# print("Accuracy of SVC is :", {accuracy_score(Ytest, Ypredict)})

# Ypredict = clf_lr.predict(Xtest)
# print("Accuracy of Logistical Regression is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_MLP.predict(Xtest)
print("Accuracy of Multi Level Perception is :", {accuracy_score(Ytest, Ypredict)})

