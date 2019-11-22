library(WGCNA)
library(dplyr)
options(stringsAsFactors = False)
enableWGCNAThreads()
#Tutorial 1 https://horvath.genetics.ucla.edu/html/CoexpressionNetwork/Rpackages/WGCNA/Tutorials/FemaleLiver-01-dataInput.pdf
#tutorial 2 https://horvath.genetics.ucla.edu/html/CoexpressionNetwork/Rpackages/WGCNA/Tutorials/FemaleLiver-02-networkConstr-auto.pdf

clinData <- read.delim("~/Thesis_Research/data_clinical_patient.txt", header=TRUE, comment.char="#")

#clean data to only gene outputs
CountData <- read.delim("~/Thesis_Research/data_log2.txt", header=TRUE)
dim(CountData)
names(CountData)
CountData = select(CountData, -c(1,2))
dim(CountData)
names(CountData)

#check data dosnt have to many missing samples
gsg = goodSamplesGenes(CountData, verbose = 3)
gsg$allOK

#cluster samples
sampleTree = hclust(dist(CountData), method = "average")

#plot 
sizeGrWindow(12, 9)
par(cex = 0.6)
par(mar = c(0,4,2,0))
plot(sampleTree, main = "Sample Clustering of Outliers", sub="", xlab ="", cex.lab =1.5, cex.axis =1.5, cex.main =2)

#set soft threshold
powers = c(c(1:10), seq(from = 12, to =20, by =2))
#call the network analysis function
sft = pickSoftThreshold(CountData, powerVector = powers, verbose = 5)

#pick results=
sizeGrWindow(9,5)
par(mfrow = c(1,2))
cex1 = 0.9

#Scale-free topology fit index as a function of the soft-thresholding power
plot(sft$fitIndicies[,1], -sign(sft$fitIndicies[,3])*sft$fitIndices[,2], 
      xlab="Soft Threshold (power)",ylab="Scale Free Topology Model Fit,signed R^2",
     type="n",main = paste("Scale independence"))
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],labels=powers,cex=cex1,col="red")

# this line corresponds to using an R^2 cut-off of h
abline(h=0.90,col="red")

# Mean connectivity as a function of the soft-thresholding power
plot(sft$fitIndices[,1], sft$fitIndices[,5],xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",main = paste("Mean connectivity"))
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1,col="red")


#Net Creation 
net = blockwiseModules(CountData, power = 6,
                       TOMType = "unsigned", minModuleSize = 30,
                       reassignThreshold = 0, mergeCutHeight = 0.25,
                       numericLabels = TRUE, pamRespectsDendro = FALSE,
                       saveTOMs = TRUE,
                       saveTOMFileBase = "femaleMouseTOM",
                       verbose = 3)

#Net Graph 
sizeGrWindow(12, 9)
mergedColors = labels2colors(net$colors)

plotDendroAndColors(net$dendrograms[[1]], mergedColors[net$blockGenes[[1]]],"Module colors",dendroLabels = FALSE, hang = 0.03,addGuide = TRUE, guideHang = 0.05)
moduleLabels = net$colors
moduleColors = labels2colors(net$colors)
MEs = net$MEs
geneTree = net$dendrograms[[1]]
save(MEs, moduleLabels, moduleColors, geneTree,file = "CountData_autoNetworkConstruction.RData")