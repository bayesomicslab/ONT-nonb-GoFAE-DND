if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("IWTomics")

library(IWTomics)

###################################################################################################################
## For now some of our data is ready

data_path <- 'Data/iwt_10000'
plot_path <- 'Data/iwt_10000/plots'
datasets=read.table(file.path(data_path,"datasets.txt"), sep="\t",header=TRUE,stringsAsFactors=FALSE)
features_datasetsTable=read.table(file.path(data_path,"features_datasetsTable.txt"), sep="\t",header=TRUE,stringsAsFactors=FALSE)
regionsFeatures=IWTomics::IWTomicsData(datasets$RegionFile,features_datasetsTable[,3:10],'center', datasets$id,datasets$name,
                                             features_datasetsTable$id,features_datasetsTable$name, path=file.path(data_path,'files'))
save(regionsFeatures,file=paste0(file.path(data_path, "data_two_measurements.RData"))


# quantile test on each feature  and region versus control
quantile_regionsFeatures_test_pairs=IWTomicsTest(regionsFeatures,id_region1=c('A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat',
                                                                              'Mirror_Repeat', 'Direct_Repeat', 'Short_Tandem_Repeat',
                                                                              'Z_DNA_Motif'),
                                                 id_region2=c('Control', 'Control', 'Control', 'Control', 'Control', 'Control', 'Control'),
                                                 statistics='quantile',probs=c(0.05,0.25,0.5,0.75,0.95),B=10000)
save(quantile_regionsFeatures_test_pairs,file=paste0(file.path(data_path, "iwt_results_10000_quantile_pairs.RData")))




# median statistics test on each feature  and region versus control
mean_regionsFeatures_test_pairs=IWTomicsTest(regionsFeatures,id_region1=c('A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat',
                                                                             'Mirror_Repeat', 'Direct_Repeat', 'Short_Tandem_Repeat',
                                                                             'Z_DNA_Motif'),
                                          id_region2=c('Control', 'Control', 'Control', 'Control', 'Control', 'Control', 'Control'),
                                          statistics='median',B=10000)
save(mean_regionsFeatures_test_pairs,file=paste0(file.path(data_path, "iwt_results_10000_median_pairs.RData")))



## Adjusted p-value for each comparison and each feature
adjusted_pval(mean_regionsFeatures_test_pairs)


plotTest(mean_regionsFeatures_test_pairs)


## Summary plot of the two sample tests
pdf(file.path(plot_path, "Summary_plot.pdf"))
plotSummary(quantile_regionsFeatures_test_pairs,groupby='feature',align_lab='Center')
dev.off()
