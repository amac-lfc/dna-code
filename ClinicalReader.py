import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Clin = pd.read_csv("../../FireBrowse/Clinical_Level/OV.clin.merged.txt", \
                    delimiter = "\t")
Clin = Clin.to_numpy()
Exp = pd.read_csv("../../FireBrowse/OV.normalization_gene__data.data.txt", \
                    delimiter = "\t")


# Days to death
DtD = Clin[22]
y = [x for x in DtD if str(x)!= 'nan']
y = y[1:]
y = np.asarray(y)
y = y.astype('int')
x = np.arange(0, len(y), 1)


# number survied and died
life = Clin[736]
life = life[1:]
# count = np.in1d(life, 'alive').sum()

# plt.hist(y, bins=[0,1115, 2250, 3375, 4500])
# plt.show()

# # Scatter plot ofo days survived
# plt.scatter(x, y)
# plt.xlabel("patient")
# plt.ylabel("Days to Death")
# # plt.yticks(np.arange(0, 3000, 100))
# plt.show()


# Patient ID
CPat = Clin[17]

Etemp = Exp.columns
Epat = np.zeros(0)
for i in range(len(Etemp)):
    np.append(Epat, Etemp[i])
print(Epat)
