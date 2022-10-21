if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("IWTomics")
library(IWTomics)
data_path <- D:/UCONN/nonBDNA/Data/iwt_1000
datasets=read.table(file.path(D:/UCONN/nonBDNA/Data/iwt_1000,"datasets.txt"), sep="	",header=TRUE,stringsAsFactors=FALSE)
features_datasetsTable=read.table(file.path(D:/UCONN/nonBDNA/Data/iwt_1000,"features_datasetsTable.txt"), sep="	",header=TRUE,stringsAsFactors=FALSE)
regionsFeatures=IWTomics::IWTomicsData(datasets$RegionFile,features_datasetsTable[,3:10],"center", datasets$id,datasets$name, features_datasetsTable$id,features_datasetsTable$name, path=file.path(D:/UCONN/nonBDNA/Data/iwt_1000,"files"))
save(regionsFeatures,file=paste0(D:/UCONN/nonBDNA/Data/iwt_1000\rdata))
