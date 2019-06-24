import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

ClinD = pd.read_csv("../../FireBrowse/Clinical_Level/OV.clin.merged.txt", \
                    delimiter = "\t")
Clin = ClinD.to_numpy()


#stage cancer data fetching and histogram
stages = Clin[822][1:582]

# print(stages)
# stageList = np.empty(len(stages), dtype = object)
# for i in range(len(stages)):
#     temp = stages[i]
#     print(temp)
#     if str(temp) != 'nan':
#         temp2 = temp[6:]
#         stageList[i] = temp2
# print(stageList)

# stageCounter = Counter(stageList)
# tempdf = pd.DataFrame.from_dict(stageCounter, orient = 'index')
# tempdf.plot(kind = 'bar')
# plt.savefig("Cancer Stage")
# plt.show()
# plt.clf()
#More Data

# RadList = Clin[780]
# ChemoList = Clin[19]
# SurgeryList = Clin[3456]
# HormonalTherapy = Clin[741]
# ImmunoTherapy = Clin[745]
# HistTreatList = Clin[3482]
Age = Clin[14][1:582]


#histogram of age 
print(Age)
Age = Age.astype(int)
plt.hist(Age, bins = [20,30,40,50,60,70,80,90,100])
plt.savefig("Age Histogram  ")
plt.show()