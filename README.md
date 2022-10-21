# ONT-nonb-GoFAE-COD
Goodness-of-fit contrastive autoencoder for outlier detection of non-B DNA in ONT sequencing.

**NOTE:** **_GoF code will be released upon acceptance._**

## What is non-B DNA?

Non-canonical (or non-B) DNA are genomic regions where the three-dimensional conformation of
DNA deviates from the double helical structure. Non-B DNA play an important role in basic cellular
processes and have been associated with genomic instability, gene regulation, and oncogenesis.
The DNA conformations that we study here are the following:


![plot](./figures/dna_conformation.PNG?raw=true)




## Reads Processing:

#### Albacore Basecalling:

```Albacore
~$ read_fast5_basecaller.py -f FLO-PRO002 -k SQK-LSK109 --input $path/na12878/fast5/single/ --save_path $path/na12878/fast5/albacore_single/ --output_format fastq,fast5 -t 48 --recursive --config r941_450bps_linear_prom.cfg
```
#### Tombo re-squiggle:

```Tombo
~$ tombo resquiggle $path/workspace/pass/ hg38.fa --dna --overwrite --basecall-group Basecall_1D_001 --include-event-stdev --failed-reads-filename $path/workspace/pass/tombo_failed_reads.txt --processes 48
```






## Prepare the input:

### Step 1:
Extract motifs postions from non B DNA DB.

### Step 2: 
Fix windows of length x around motifs.

### Step 3: 
Extend the positions on the opposite strand.

### Step 4: 
Find high quality reads that fall on the windows.

### Step 5:
Find motif free regions.

### Step 6:
Compute translocation signal on the non-overlapping windows.

