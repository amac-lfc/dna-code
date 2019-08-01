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

# aliveTrain = np.where(training[:,-1] == 1)[0]
# deadTrain = np.where(training[:,-1] == 0)[0]
# aliveTest = np.where(testing[:,-1] == 1)[0]
# deadTest = np.where(testing[:,-1] == 0)[0]

#making the Classifiers
clf_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

#Accuracy Score Array:
AccuracyScoreArray = np.empty((len(testing[:, 0]),2))        

for i in range(int((len(np.where(testing[:,-1] == float(1))[0]+1)))):                                                                                                                                                     
    #shuffle
    np.random.shuffle(training)
    np.random.shuffle(testing)


    #train and test datasets    
    testingInt = testing.astype(int)
    trainingInt = training.astype(int)
    Ytrain = trainingInt[:, -1]
    Xtrain = training[:, :-1]

    index0 = np.where(testing[:,-1] == 0)[0]
    index1 = np.where(testing[:,-1] == 1)[0]
    indexDead = np.append(index0, index1[:i])
    indexAlive = np.append(index1, index0[:i])

    indexDead = indexDead.astype(int)
    indexAlive = indexAlive.astype(int)

    YtestDead = testing[indexDead, -1]
    XtestDead = testing[indexDead, :-1]

    YtestAlive = testing[indexAlive, -1]
    XtestAlive = testing[indexAlive, :-1]


    #training
    clf_RF = clf_RF.fit(Xtrain, Ytrain)

    #alive dead percentages

    aliveTrain = np.where(Ytrain == 1)[0]
    deadTrain = np.where(Ytrain == 0)[0]
    aliveTest = np.where(YtestDead == 1)[0]
    deadTest = np.where(YtestDead == 0)[0]

    #First Test
    print('\n')
    print('First Test')
    print('Loop Number: ', i+1)
    print("Percent of patients alive TEST",len(aliveTest)/len(YtestDead))
    print("Percent of paitents dead TEST", len(deadTest)/len(YtestDead))

    print("Percent of patients alive TRAIN",len(aliveTrain)/len(Ytrain))
    print("Percent of paitents dead TRAIN", len(deadTrain)/len(Ytrain))

    # num of features
    print('Number of Features : ', len(features))
    print('Ytrain:', Ytrain)

    #prediction results\
    print("\n")
    Ypredict = clf_RF.predict(XtestDead)
    conf_matrix = metrics.confusion_matrix(YtestDead, Ypredict)

    print("Accuracy of Random Forest is :", {accuracy_score(YtestDead, Ypredict)})
    if i == 0:
        AccuracyScoreArray[i] = [0, accuracy_score(YtestDead, Ypredict)]
    else:
        AccuracyScoreArray[i] = [metrics.roc_auc_score(YtestDead, Ypredict), accuracy_score(YtestDead, Ypredict)]
    print("True Positive:" ,conf_matrix[0][0])
    print(conf_matrix[0][0]/len(YtestDead))
    print("False Positive:", conf_matrix[0][1])
    print(conf_matrix[0][1]/len(YtestDead))
    print("True Negative:", conf_matrix[1][1])
    print("False Negative:", conf_matrix[1][0])
    print('\n')

    #Second test

    aliveTest = np.where(YtestAlive == 1)[0]
    deadTest = np.where(YtestAlive == 0)[0]

    print('\n')
    print('Second Test')
    print('Loop Number: ', i+1)
    print("Percent of patients alive TEST",len(aliveTest)/len(YtestAlive))
    print("Percent of paitents dead TEST", len(deadTest)/len(YtestAlive))

    print("Percent of patients alive TRAIN",len(aliveTrain)/len(Ytrain))
    print("Percent of paitents dead TRAIN", len(deadTrain)/len(Ytrain))

    # num of features
    print('Number of Features : ', len(features))
    print('Ytrain:', Ytrain)

    #prediction results\
    print("\n")
    Ypredict = clf_RF.predict(XtestAlive)
    conf_matrix = metrics.confusion_matrix(YtestAlive, Ypredict)

    print("Accuracy of Random Forest is :", {accuracy_score(YtestAlive, Ypredict)})
    if i == 0:
        AccuracyScoreArray[len(AccuracyScoreArray)-(i+1)] = [0, accuracy_score(YtestAlive, Ypredict)]
    else:
        AccuracyScoreArray[len(AccuracyScoreArray)-(i+1)] = [metrics.roc_auc_score(YtestAlive, Ypredict), accuracy_score(YtestAlive, Ypredict)]
    print("True Positive:" ,conf_matrix[0][0])
    print(conf_matrix[0][0]/len(YtestAlive))
    print("False Positive:", conf_matrix[0][1])
    print(conf_matrix[0][1]/len(YtestAlive))
    print("True Negative:", conf_matrix[1][1])
    print("False Negative:", conf_matrix[1][0])
    print('\n')

np.save('AccuracyScoreArray', AccuracyScoreArray, allow_pickle=True)