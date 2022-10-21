echo `hostname`
module load albacore;
source="/labs/Aguiar/non_bdna/human/na12878/fast5/single";
dest="/labs/Aguiar/non_bdna/human/na12878/fast5/dist_albacore_single";
for n in {0..226}; do read_fast5_basecaller.py -f FLO-PRO002 -k SQK-LSK109 --input $source/$n/ --save_path $dest/$n/ --output_format fastq,fast5 -t 48 --recursive --config r941_450bps_linear_prom.cfg; echo "$n basecalling is done"; done
