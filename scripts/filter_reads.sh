module load samtools
module load bedtools

# For chromosomes in positive strand
samtools view merged.bam chr1:10005-248946452 -F 0xF14 -o $filtered_path/chr1+.bam;
bedtools bamtobed -i $filtered_path/chr1+.bam > $filtered_path/chr1+.bed;

# For chromosomes in negative strand
samtools view merged.bam chr1:10005-248946452 -F 0xF04 -f 0x10 -o $filtered_path/chr1-.bam;
bedtools bamtobed -i $filtered_path/chr1-.bam > $filtered_path/chr1-.bed;
