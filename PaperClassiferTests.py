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
# def getImportances(classifier, X, features_list, targetFile):
#     importances = classifier.feature_importances_
#     std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
#                  axis=0)
#     indices = np.argsort(importances)[::-1]

#     # Print the feature ranking and save in file
#     print("Feature ranking:")
#     t = open(targetFile, "w+")

#     for f in range(X.shape[1]):
#         t.write(f"{f + 1}. Feature {features_list[indices[f]]} ({importances[indices[f]]})"+"\n")

# def getImportancesTrees(classifier, X, features_list, targetFile):
#     importances = classifier.feature_importances_
#     indices = np.argsort(importances)[::-1]

#     # Print the feature ranking and save in file
#     print("Feature ranking:")
#     t = open(targetFile, "w+")

#     for f in range(X.shape[1]):
#         t.write(f"{f + 1}. Feature {features_list[indices[f]]} ({importances[indices[f]]})"+"\n")


#file imports
training = np.load("PapersDataSet/sample_training_replacement_mean.npy")
for i in range(training.shape[0]):
    if str(training[i, -1]) == 'Up':
        training[i, -1] = 1
    if str(training[i, -1]) == 'Down':
        training[i, -1] = 0
training = training.astype(float)

testing = np.load("PapersDataSet/sample_testing_replacement_mean.npy")
for i in range(testing.shape[0]):
    if str(testing[i, -1]) == 'Up':
        testing[i, -1] = 1
    if str(testing[i, -1]) == 'Down':
        testing[i, -1] = 0
testing = testing.astype(float)

features = np.load("PapersDataSet/sample_features.npy")

aliveTrain = np.where(training[:,-1] == 1)[0]
deadTrain = np.where(training[:,-1] == 0)[0]
aliveTest = np.where(testing[:,-1] == 1)[0]
deadTest = np.where(testing[:,-1] == 0)[0]

#Feature List


#shuffle
np.random.shuffle(training)
np.random.shuffle(testing)

# where0 = np.where(train[:, -1]==0)[0]
# whereQ = np.where(train[:, -1]==1)[0]
# for k in range(int(len(whereQ)-len(where0))):
#     temp = np.random.choice(where0)
#     train = np.vstack((train, train[temp,:]))

#train and test datasets
Xtest = testing[:, :-1]
Ytest = testing[:, -1]
Xtrain = training[:, :-1]
Ytrain = training[:, -1]

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

clf_MLP = MLPClassifier(solver='sgd', hidden_layer_sizes=(20,2), shuffle=True, batch_size=25, learning_rate='adaptive')

#training
clf_RF = clf_RF.fit(Xtrain, Ytrain)
clf_tree = clf_tree.fit(Xtrain, Ytrain)
clf_svm = clf_svm.fit(Xtrain, Ytrain)
clf_lr = clf_lr.fit(Xtrain, Ytrain)
clf_MLP = clf_MLP.fit(Xtrain, Ytrain)

# getImportances(clf_RF, Xtrain, Features, 'RandomForestFeatures.txt')
# getImportancesTrees(clf_tree, Xtrain, Features, "TreeFeatures_"+str(top)+ '.txt')
# getImportances(clf_svm, Xtrain, Features, "SVMFeatures_"+str(top)+ '.txt')

#alive dead percentages
print("Percent of patients alive TEST",len(aliveTest)/len(Ytest))
print("Percent of paitents dead TEST", len(deadTest)/len(Ytest))

print("Percent of patients alive TRAIN",len(aliveTrain)/len(Ytrain))
print("Percent of paitents dead TRAIN", len(deadTrain)/len(Ytrain))

# num of features
print('Number of Features : ', len(features))


#prediction results
Ypredict = clf_RF.predict(Xtest)
# print("Prediction",Ypredict)
# print("Actual",Ytest)
print("Accuracy of Random Forest is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_tree.predict(Xtest)
# print("Prediction",Ypredict)
# print("Actual",Ytest)
print("Accuracy of Trees is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_svm.predict(Xtest)
# print("Prediction",Ypredict)
# print("Actual",Ytest)
print("Accuracy of SVC is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_lr.predict(Xtest)
# print("Prediction",Ypredict)
# print("Actual",Ytest)
print("Accuracy of Logistical Regression is :", {accuracy_score(Ytest, Ypredict)})

Ypredict = clf_MLP.predict(Xtest)
# print("Prediction",Ypredict)
# print("Actual",Ytest)
print("Accuracy of  Multi Layer Perceptron is :", {accuracy_score(Ytest, Ypredict)})