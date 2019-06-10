import pandas as pd

Clin = pd.read_csv("../../FireBrowse/Clinical_Level/OV.clin.merged.txt", \
                    delimiter = "\t")

DtD = Clin[Clin["admin.batch_number"].str.match("patient.days_to_death")]
print(DtD)
