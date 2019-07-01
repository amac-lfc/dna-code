import numpy as np
import numpy.core.defchararray as np_f

RF = np.loadtxt("TotalImportanceTopQuintilesauto.txt", dtype=str)
GElist = np.load("FirePkl/GElist.npy")

#GElist = np.append(GElist[:], np.array(['Age', 'Stage', 'Treatment']))

RF = np_f.replace(RF, "['", "")
RF = np_f.replace(RF, '["', "")
RF = np_f.replace(RF, "]", "")
RF = [x[:-1] for x in RF[:,0]]

Interface = np.zeros(len(RF)-3)
k=0
i=0
while i < (len(RF)-3):
        if (RF[k] == 'Treatment' or RF[k] == 'Age' or RF[k] == 'Stage'):
                k = k+1
        else:
                temp = (np.where(RF[k] == GElist)[0])
                print(RF[k])
                print(temp)
                Interface[i] = temp[0]
                k += 1
                i += 1
        # print(i)

print(Interface)
np.save("FirePkl/RFTopInterfaceQuintiles", Interface, allow_pickle=True)