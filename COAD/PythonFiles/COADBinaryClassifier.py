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
Y = np.load("FirePkl/COAD_OutlierRemovedY.npy")
Y = Y.astype(int)
X = np.load("FirePkl/COAD_OutlierRemovedX.npy")
X = X.astype(float)
GElist = np.load("FirePkl/COAD_OutlierRemovedGenes.npy")

alive = np.where(Y == 1)[0]
dead = np.where(Y == 0)[0]

Data = np.column_stack((X, Y))

# index = Y == -1
# Y[index] = 0
# index = Y > 0
# Y[index] = 1

#Feature List
# Features = np.append(GElist[RFTopInterface[:top]], np.array(['Age', 'Stage', 'Treatment']))
Features = np.copy(GElist)

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

#train and test datasets
Xtest = test[:, :-1]
Ytest = test[:, -1]
Xtrain = train[:, :-1]
Ytrain = train[:, -1]

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
clf_RF = clf_RF.fit(Xtrain, Ytrain)
clf_tree = clf_tree.fit(Xtrain, Ytrain)
clf_svm = clf_svm.fit(Xtrain, Ytrain)
clf_lr = clf_lr.fit(Xtrain, Ytrain)

getImportances(clf_RF, Xtrain, Features, "RandomForestFeatures_"+str(top)+ '.txt')
getImportancesTrees(clf_tree, Xtrain, Features, "TreeFeatures_"+str(top)+ '.txt')
# getImportances(clf_svm, Xtrain, Features, "SVMFeatures_"+str(top)+ '.txt')

#alive dead percentages
alive = np.where(Y == 1)[0]
dead = np.where(Y == 0)[0]
print("Percent of patients alive",len(alive)/len(Y))
print("Percent of paitents dead", len(dead)/len(Y))
alive = np.where(Ytest == 1)[0]
dead = np.where(Ytest == 0)[0]
print("Percent of patients alive TEST",len(alive)/len(Ytest))
print("Percent of paitents dead TEST", len(dead)/len(Ytest))
alive = np.where(Ytrain == 1)[0]
dead = np.where(Ytrain == 0)[0]
print("Percent of patients alive TRAIN",len(alive)/len(Ytrain))
print("Percent of paitents dead TRAIN", len(dead)/len(Ytrain))

# num of features
print('Number of Features : ', len(GElist))


#prediction results
Ypredict = clf_RF.predict(Xtest)
print("Prediction",Ypredict)
print("Actual",Ytest)
print("Accuracy of Random Forest is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_tree.predict(Xtest)
print("Prediction",Ypredict)
print("Actual",Ytest)
print("Accuracy of Trees is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_svm.predict(Xtest)
print("Prediction",Ypredict)
print("Actual",Ytest)
print("Accuracy of SVC is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_lr.predict(Xtest)
print("Prediction",Ypredict)
print("Actual",Ytest)
print("Accuracy of Logistical Regression is :", {accuracy_score(Ytest, Ypredict)})
