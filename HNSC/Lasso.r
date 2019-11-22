library(glmnet)
library(readxl)
library(DESeq2)
library(dplyr)
library(glmmLasso)
library(mvtnorm)

#load Data Sets
load("./filteredData.Rdata")
load("./filteredClin.Rdata")
Clin <- filteredClin
GeneData <-filteredData

#setting up Y
Y <- Clin$`Overall Survival Status`
Y[Y=="DECEASED"] <- 0
Y[Y=="LIVING"] <- 1

#setting up the features
tempData = data[,-1]

glm.obj <- glmmLasso(pain ~ time + th + age + sex, rnd = NULL,family = cumulative(), data = knee, lambda=10,switch.NR=TRUE, control=list(print.iter=TRUE))
