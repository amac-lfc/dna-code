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

RadList = Clin[780]
print(RadList, len(RadList))
ChemoList = Clin[566][1:582]
print(ChemoList, len(ChemoList))
SurgeryList = Clin[3456]
HormonalTherapy = Clin[741]
print(HormonalTherapy, len(HormonalTherapy))
ImmunoTherapy = Clin[745]
print(ImmunoTherapy, len(ImmunoTherapy))
HistTreatList = Clin[3482]
Age = Clin[14][1:582]

from collections import Counter
a = ['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'd', 'e', 'e', 'e', 'e', 'e']
Treatment_counts = Counter(ChemoList)
df = pd.DataFrame.from_dict(Treatment_counts, orient='index')
df.plot(kind='bar')
plt.savefig("treatment type")
plt.show()

# print(len(Age), len(stages))
# #histogram of age 
# print(Age)
# Age = Age.astype(int)
# plt.hist(Age, bins = [20,30,40,50,60,70,80,90,100])
# plt.savefig("Age Histogram  ")
# plt.show()