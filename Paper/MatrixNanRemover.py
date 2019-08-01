import pandas as pd
import numpy as np

training = pd.read_csv("Pickles/sample_training_nofeatureselection.csv")
testing = pd.read_csv("Pickles/sample_testing_nofeatureselection.csv")

training = training.replace('Down', 0)
training = training.replace('Up', 1)
testing = testing.replace('Down', 0)
testing = testing.replace('Up', 1)

training = training.drop(columns = [training.columns[0]])
testing = testing.drop(columns = [testing.columns[0]])

training2 = training.replace('?', 99)
testing2 = testing.replace('?', 99)

training2 = training2.astype(float)
testing2 = testing2.astype(float)
np.save('Pickles/sample_training_full_replacement99', training2, allow_pickle=True)
np.save('Pickles//sample_testing_full_replacment99', testing2, allow_pickle=True)

training = training.replace('?', np.nan)
testing = testing.replace('?', np.nan)

trainMean = training.mean(axis = 1, skipna = True)
testMean = testing.mean(axis = 1, skipna = True)
trainstd = training.std(axis = 1, skipna = True)
teststd = testing.std(axis = 1, skipna = True)

for i in training.columns:
    k = 0
    training[i] = training[i].replace(np.nan, trainMean[k]+trainstd[k]*np.random.uniform(-1,1))
    k += 1

for i in testing.columns:
    k = 0
    testing[i] = testing[i].replace(np.nan, testMean[k]+teststd[k]*np.random.uniform(-1,1))
    k += 1

training = training.astype(float)
testing = testing.astype(float)
print(training.shape)
print(testing.shape)
np.save("Pickles/sample_training_full_replacement_mean", training, allow_pickle=True)
np.save('Pickles/sample_testing_full_replacement_mean', testing, allow_pickle=True)


Features = training.columns
np.save("Pickles/sample_features", Features, allow_pickle=True)