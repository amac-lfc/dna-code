import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import random as rd

#true vs false positive rate

#file imports
training = np.load("Pickles/sample_training_replacement_mean.npy")
for i in range(training.shape[0]):
    if str(training[i, -1]) == 'Up':
        training[i, -1] = 1
    if str(training[i, -1]) == 'Down':
        training[i, -1] = 0
training = training.astype(float)

testing = np.load("Pickles/sample_testing_replacement_mean.npy")
for i in range(testing.shape[0]):
    if str(testing[i, -1]) == 'Up':
        testing[i, -1] = 1
    if str(testing[i, -1]) == 'Down':
        testing[i, -1] = 0
testing = testing.astype(float)

print(training.shape)
print(testing.shape)
features = np.load("Pickles/sample_features.npy")

aliveTrain = np.where(training[:,-1] == 1)[0]
deadTrain = np.where(training[:,-1] == 0)[0]
aliveTest = np.where(testing[:,-1] == 1)[0]
deadTest = np.where(testing[:,-1] == 0)[0]

#shuffle
np.random.shuffle(training)
np.random.shuffle(testing)


#train and test datasets
Xtest = testing[:, :-1]
Ytest = testing[:, -1]
Ytest = Ytest.astype(int)
Xtrain = training[:, :-1]
Ytrain = training[:, -1]
Ytrain = Ytrain.astype(int)

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

clf_MLP = MLPClassifier(solver='sgd', \
            shuffle=True, batch_size='auto', learning_rate='adaptive')

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
print('\n')
print("Percent of patients alive TEST",len(aliveTest)/len(Ytest))
print("Percent of paitents dead TEST", len(deadTest)/len(Ytest))

print("Percent of patients alive TRAIN",len(aliveTrain)/len(Ytrain))
print("Percent of paitents dead TRAIN", len(deadTrain)/len(Ytrain))

# num of features
print('Number of Features : ', len(features))
print('Ytrain:', Ytrain)

#prediction results\
print("\n")
Ypredict = clf_RF.predict(Xtest)
conf_matrix = metrics.confusion_matrix(Ytest, Ypredict)

print("AUC of Random Forest is :", {metrics.roc_auc_score(Ytest, Ypredict)})
print("Accuracy of Random Forest is :", {accuracy_score(Ytest, Ypredict)})
print("True Positive:" ,conf_matrix[0][0])
print(conf_matrix[0][0]/len(Ytest))
print("False Positive:", conf_matrix[0][1])
print(conf_matrix[0][1]/len(Ytest))
print("True Negative:", conf_matrix[1][1])
print("False Negative:", conf_matrix[1][0])
print('\n')


Ypredict = clf_tree.predict(Xtest)
conf_matrix = metrics.confusion_matrix(Ytest, Ypredict)

print("AUC of Trees is :", {metrics.roc_auc_score(Ytest, Ypredict)})
print("Accuracy of Trees is :", {accuracy_score(Ytest, Ypredict)})
print("True Positive:" ,conf_matrix[0][0])
print("False Positive:", conf_matrix[0][1])
print("True Negative:", conf_matrix[1][1])
print("False Negative:", conf_matrix[1][0])
print('\n')

Ypredict = clf_svm.predict(Xtest)
conf_matrix = metrics.confusion_matrix(Ytest, Ypredict)

print("AUC of SVC is :", {metrics.roc_auc_score(Ytest, Ypredict)})
print("Accuracy of SVC is :", {accuracy_score(Ytest, Ypredict)})
print("True Positive:" ,conf_matrix[0][0])
print("False Positive:", conf_matrix[0][1])
print("True Negative:", conf_matrix[1][1])
print("False Negative:", conf_matrix[1][0])
print('\n')

Ypredict = clf_lr.predict(Xtest)
conf_matrix = metrics.confusion_matrix(Ytest, Ypredict)

print("AUC of Logistical regression is :", {metrics.roc_auc_score(Ytest, Ypredict)})
print("Accuracy of Logistical Regression is :", {accuracy_score(Ytest, Ypredict)})
print("True Positive:" ,conf_matrix[0][0])
print("False Positive:", conf_matrix[0][1])
print("True Negative:", conf_matrix[1][1])
print("False Negative:", conf_matrix[1][0])
print('\n')

Ypredict = clf_MLP.predict(Xtest)
conf_matrix = metrics.confusion_matrix(Ytest, Ypredict)

print("AUC of Multi Layer Perceptron is :", {metrics.roc_auc_score(Ytest, Ypredict)})
print("Accuracy of  Multi Layer Perceptron is :", {accuracy_score(Ytest, Ypredict)})
# print(Ypredict, Ytest)
print("True Positive:" ,conf_matrix[0][0])
print("False Positive:", conf_matrix[0][1])
print("True Negative:", conf_matrix[1][1])
print("False Negative:", conf_matrix[1][0])
print('\n') 