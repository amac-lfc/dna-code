import numpy as np
import numpy.core.defchararray as np_f

RF = np.loadtxt("RFTopWeight.txt", dtype=str)
GElist = np.load("FirePkl/GElist.npy")

GElist = np.append(GElist[:], np.array(['Age', 'Stage', 'Treatment']))

RF = np_f.replace(RF, "[", "")
RF = np_f.replace(RF, "'", "")
RF = np_f.replace(RF, "]", "")

print(RF.shape)

Interface = np.zeros(len(RF[:,0]))
for i in range(len(RF)):
    temp = (np.where(RF[:, 0][i] == GElist)[0])
    if len(temp) > 0:
        Interface[i] = temp[0]


np.save("FirePkl/RFTopInterface", Interface, allow_pickle=True)