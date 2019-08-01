import pandas as pd

METc =  pd.read_csv("METcancer_OV.csv", delimiter = ",")
GEc  = pd.read_csv("GEcancer_OV.csv", delimiter = ",")

METn =  pd.read_csv("METnormal_OV.csv", delimiter = ",")
GEn = pd.read_csv("GEnormal_OV.csv", delimiter = ",")


METc.to_pickle("METcancerPickle.pkl")
GEc.to_pickle("GEcancerPickle.pkl")

METn.to_pickle("METnormalPickle.pkl")
GEn.to_pickle("GEnormalPickle.pkl")
