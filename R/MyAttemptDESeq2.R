if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("DESeq2")

HRNA <- read.csv('DataUsedforDESeq2/HNSC.rnaseq__illuminahiseq_rnaseq__unc_edu__Level_3__gene_expression__data.data.txt', header = TRUE, row.names = 1)
HRNA_data <- data.matrix(HRNA)

dds <- HRNA_data(countData = cts,
                              colData = coldata,
                              design= ~ batch + condition)
dds <- DESeq(dds)
resultsNames(dds) # lists the coefficients
res <- results(dds, name="condition_trt_vs_untrt")
# or to shrink log fold changes association with condition:
res <- lfcShrink(dds, coef="condition_trt_vs_untrt", type="apeglm")