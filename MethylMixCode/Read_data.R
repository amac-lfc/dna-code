data_OV <- readRDS("/home/bousquet/Documents/Lake Forest/DNA data/data_OV.rds")

write.csv(data_OV[["METcancer"]], file = "METcancer_OV.csv",row.names=TRUE)
write.csv(data_OV[["GEcancer"]], file = "GEcancer_OV.csv",row.names=TRUE)
write.csv(data_OV[["METnormal"]], file = "METnormal_OV.csv",row.names=TRUE)
write.csv(data_OV[["GEnormal"]], file = "GEnormal_OV.csv",row.names=TRUE)
write.csv(data_OV[["ProbeMapping"]], file = "ProbeMapping_OV.csv",row.names=TRUE)