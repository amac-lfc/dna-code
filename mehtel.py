import pandas as pd
import numpy as np

ClinD = pd.read_csv("../../FireBrowse/Clinical_Level/OV.clin.merged.txt", \
                    delimiter = "\t")
Clin = ClinD.to_numpy()


stages = Clin[822][1:582]
print(stages)
stageList = []
for i in range(len(stages)):
    temp = stages[i]
    stageList.append(temp[6:])
print(stageList)

# stage = np.empty(len(stages), dtype = object)
# for i in range(len(stages)):
#     temp = stages[i]
#     stage[i] = str(temp[i][6:])
# print(stage)




#More Data

RadList = Clin[780]
ChemoList = Clin[19]
SurgeryList = Clin[3456]
HormonalTherapy = Clin[741]
ImmunoTherapy = Clin[745]
HistTreatList = Clin[3482]
Age = Clin[14]
