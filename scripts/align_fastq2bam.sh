module load minimap2/2.22
module load samtools
minimap2 -ax map-ont hg38.fna merged.fastq | samtools sort -@5 -o merged.bam;
samtools index merged.bam;
