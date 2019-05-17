library(MethylMix)
library(foreach)
library(iterators)
library(parallel)
library(doParallel)

cancerSite <- "COAD" #LUSC and LUAD (our test is COAD)
setwd("~/Documents/Lake\ Forest/DNA\ data")
targetDirectory <- paste0(getwd(), "/")

cl <- makeCluster(20)
registerDoParallel(cl)

# Downloading methylation data
METdirectories <- Download_DNAmethylation(cancerSite, targetDirectory)
# Processing methylation data
METProcessedData <- Preprocess_DNAmethylation(cancerSite, METdirectories)
# Saving methylation processed data
saveRDS(METProcessedData, file = paste0(targetDirectory, "MET_", cancerSite, 
                                        "_Processed.rds"))

# Downloading gene expression data
GEdirectories <- Download_GeneExpression(cancerSite, targetDirectory)
# Processing gene expression data
GEProcessedData <- Preprocess_GeneExpression(cancerSite, GEdirectories)
# Saving gene expression processed data
saveRDS(GEProcessedData, file = paste0(targetDirectory, "GE_", cancerSite, "_Processed.rds"))

# Clustering probes to genes methylation data
METProcessedData <- readRDS(paste0(targetDirectory, "MET_", cancerSite, "_Processed.rds"))
res <- ClusterProbes(METProcessedData[[1]], METProcessedData[[2]])

# Putting everything together in one file
toSave <- list(METcancer = res[[1]], METnormal = res[[2]], GEcancer = GEProcessedData[[1]], 
               GEnormal = GEProcessedData[[2]], ProbeMapping = res$ProbeMapping)
saveRDS(toSave, file = paste0(targetDirectory, "data_", cancerSite, ".rds"))

stopCluster(cl)

# cancerSite <- "OV"
# targetDirectory <- paste0(getwd(), "/")
# 
# library(doParallel)
# cl <- makeCluster(5)
# registerDoParallel(cl)
# GetData(cancerSite, targetDirectory)
# stopCluster(cl)