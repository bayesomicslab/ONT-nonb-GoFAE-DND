#if (!require("BiocManager", quietly = TRUE))
#  install.packages("BiocManager",lib='~/bin/Rpackages')

#BiocManager::install("IWTomics", lib='~/bin/Rpackages')
if (!dir.exists("./library")) {
 dir.create("./library")
}
old_libraries <- .libPaths()
.libPaths(c('~/bin/Rpackages', old_libraries))
.libPaths()



library(IWTomics)

# paths
main_path <- '/labs/Aguiar/non_bdna/iwt/IWT_data_req5_mean/iwt_10000'
plot_path <- '/labs/Aguiar/non_bdna/iwt/IWT_data_req5_mean/iwt_10000/plots'
data_path <- '/labs/Aguiar/non_bdna/iwt/IWT_data_req5_mean/iwt_10000/data'
# data_path <- 'D:/UCONN/nonBDNA/Data'


# read datasets:
print("Read and save the data ...") # quite nicer,
datasets=read.table(file.path(main_path,"datasets.txt"), sep="\t",header=TRUE,stringsAsFactors=FALSE)
features_datasetsTable=read.table(file.path(main_path,"features_datasetsTable.txt"), sep="\t",header=TRUE,stringsAsFactors=FALSE)

# make data object
regionsFeatures=IWTomics::IWTomicsData(datasets$RegionFile,features_datasetsTable[,3:10],'center', datasets$id,datasets$name,
                                       features_datasetsTable$id,features_datasetsTable$name, path=file.path(main_path,'files'))
# save data object
save(regionsFeatures,file=paste0(path=file.path(data_path, 'regions_data_10000.RData')))



# quantile test on each feature  and region versus control
print("IWT quantile test ...") # quite nicer,
quantile_regionsFeatures_test_pairs=IWTomicsTest(regionsFeatures,id_region1=c('A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat',
                                                                              'Mirror_Repeat', 'Direct_Repeat', 'Short_Tandem_Repeat',
                                                                              'Z_DNA_Motif'),
                                                 id_region2=c('Control', 'Control', 'Control', 'Control', 'Control', 'Control', 'Control'),
                                                 statistics='quantile',probs=c(0.05,0.25,0.5,0.75,0.95),B=10000)
# save quantile test
save(quantile_regionsFeatures_test_pairs,file=paste0(path=file.path(data_path, 'iwt_results_10000_quantile_pairs.RData')))
## Adjusted p-value for each comparison and each feature

load('D:/UCONN/nonBDNA/Data/iwt_10000/iwt_results_10000_quantile_pairs.RData')
adjusted_pval(quantile_regionsFeatures_test_pairs)

pdf(file=file.path(plot_path, 'summary_plots.pdf'), width=17, height=3)
plotSummary(quantile_regionsFeatures_test_pairs,groupby='feature',align_lab='Center')
dev.off()


pdf(file=file.path(plot_path, 'plot_test.pdf'))
plotTest(quantile_regionsFeatures_test_pairs)
dev.off()
## Summary plot of the two sample tests




# median statistics test on each feature and region versus control
print("IWT Median test ...") # quite nicer,
median_regionsFeatures_test_pairs=IWTomicsTest(regionsFeatures,id_region1=c('A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat',
                                                                          'Mirror_Repeat', 'Direct_Repeat', 'Short_Tandem_Repeat',
                                                                          'Z_DNA_Motif'),
                                             id_region2=c('Control', 'Control', 'Control', 'Control', 'Control', 'Control', 'Control'),
                                             statistics='median',B=10000)
# save median test
save(median_regionsFeatures_test_pairs,file=paste0(path=file.path(data_path, 'iwt_results_10000_median_pairs.RData')))




# mean statistics test on each feature and region versus control
print("IWT Mean test ...") # quite nicer,
mean_regionsFeatures_test_pairs=IWTomicsTest(regionsFeatures,id_region1=c('A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat',
                                                                            'Mirror_Repeat', 'Direct_Repeat', 'Short_Tandem_Repeat',
                                                                            'Z_DNA_Motif'),
                                               id_region2=c('Control', 'Control', 'Control', 'Control', 'Control', 'Control', 'Control'),
                                               statistics='mean',B=10000)
# save mean test
save(mean_regionsFeatures_test_pairs,file=paste0(path=file.path(data_path, 'iwt_results_10000_mean_pairs.RData')))




# plotTest(mean_regionsFeatures_test_pairs)
## Summary plot of the two sample tests
## x11(10,5)
# because we only have one feature I groupby test.
# plotSummary(mean_regionsFeatures_test_pairs,groupby='feature',align_lab='Center')


#plots.dir.path <- list.files(tempdir(), pattern="rs-graphics", full.names = TRUE);
#plots.png.paths <- list.files(plots.dir.path, pattern=".png", full.names = TRUE)
#file.copy(from=plots.png.paths, to="D:/UCONN/nonBDNA/Data/iwt_10000/1.png")





