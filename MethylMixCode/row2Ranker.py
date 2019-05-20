import pandas as pd

data = pd.read_csv('computed_METnormal_OV.csv', delimiter = "\t", names = ["gene", "mean", "SD"])

data = data.sort_values(by=["mean"], ascending=False)

data.to_csv('Ranked_METnormal_OV.csv')
