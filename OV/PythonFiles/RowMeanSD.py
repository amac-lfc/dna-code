import csv
import numpy as np

f = open("computed_METnormal_OV.csv", "w")

with open("METnormal_OV.csv") as csvfile:
    csvdata = csv.reader(csvfile, delimiter = ",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    next(csvdata, None)
    for row in csvdata:
        m = np.mean(row[1:])
        sd = np.std(row[1:], ddof = 1)
        f.write(row[0]+"\t"+ str(m) + "\t" + str(sd) + "\n")
    f.close()
