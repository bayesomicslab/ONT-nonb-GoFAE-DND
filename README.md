# ONT-nonb-GoFAE-DND
*Deep statistical model for predicting non-B DNA structures from ONT sequencing*

[//]: # (**NOTE:** **_GoFAE-DND code will be released upon acceptance._**)

## What is non-B DNA?

Non-canonical (or non-B) DNA are genomic regions whose three-dimensional conformation
deviates from the canonical double helix. Non-B DNA play an important role in basic cellular processes
and are associated with genomic instability, gene regulation, and oncogenesis. Experimental methods are
low-throughput and can detect only a limited set of non-B DNA structures, while computational methods
rely on non-B DNA base motifs, which are necessary but not sufficient indicators of non-B structures.
The [DNA conformations](https://user-images.githubusercontent.com/45966768/228634198-7f7b219d-f7bd-4272-86d9-ad3097a080a0.PNG) that we study here are the following:


<p align="center">

  <img width=60% height=60% src="https://user-images.githubusercontent.com/45966768/228634198-7f7b219d-f7bd-4272-86d9-ad3097a080a0.PNG">

</p>

Given the dramatic increase in genome-scale data produced using ONT platforms, and in
particular ultra-long sequencing data that supports telomere-to-telomere level genome 
assembly, we sought to develop a  parallel strategy for identifying non-B DNA structure by their 
effects on sequencing speeds (translocation times) in ONT devices 
(Ni et al., [2019](https://academic.oup.com/bioinformatics/article/35/22/4586/5474907), 
Liu et al., [2019](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-018-5372-8), 
McIntyre et al., [2019](https://www.nature.com/articles/s41467-019-08289-9), 
Stoiber et al., [2017](https://www.biorxiv.org/content/10.1101/094672v2.abstract)). 



In SMRT technology, the sequencing speed is determined by polymerization 
(Liu et al., [2018](https://genome.cshlp.org/content/28/12/1767.short)).
However, ONT devices record a measurement of current at a predefined sampling rate and then aggregate the measurements 
into *strides*, which are the smallest length of measurement accepted by the basecaller and represent a single base translocation
([An introduction to the concept of events and strides](http://simpsonlab.github.io/2015/04/08/eventalign/)).


# Instructions for downloading the data and GitHub repo

mkdir data/fast5/
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB39088-288418386_Multi_Fast5.tar
tar -xvf FAB39088-288418386_Multi_Fast5.tar ; mv Notts/* data/fast5/ ; rm FAB39088-288418386_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB39075-4246400039_Multi_Fast5.tar
tar -xvf FAB39075-4246400039_Multi_Fast5.tar ; mv UBC/* data/fast5/ ; rm FAB39075-4246400039_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB39043-3709921973_Multi_Fast5.tar
tar -xvf FAB39043-3709921973_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAB39043-3709921973_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42706-4111103328_Multi_Fast5.tar
tar -xvf FAB42706-4111103328_Multi_Fast5.tar ; mv UBC/* data/fast5/ ; rm FAB42706-4111103328_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB41174-3976885577_Multi_Fast5.tar
tar -xvf FAB41174-3976885577_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAB41174-3976885577_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42260-4177064552_Multi_Fast5.tar
tar -xvf FAB42260-4177064552_Multi_Fast5.tar ; mv UBC/* data/fast5/ ; rm FAB42260-4177064552_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42804-84744914_Multi_Fast5.tar
tar -xvf FAB42804-84744914_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAB42804-84744914_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42316-216722908_Multi_Fast5.tar
tar -xvf FAB42316-216722908_Multi_Fast5.tar ; mv Notts/* data/fast5/ ; rm FAB42316-216722908_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42205-3573838535_Multi_Fast5.tar
tar -xvf FAB42205-3573838535_Multi_Fast5.tar ; mv Notts/* data/fast5/ ; rm FAB42205-3573838535_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42561-356443753_Multi_Fast5.tar
tar -xvf FAB42561-356443753_Multi_Fast5.tar ; mv Notts/* data/fast5/ ; rm FAB42561-356443753_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42473-4179682758_Multi_Fast5.tar
tar -xvf FAB42473-4179682758_Multi_Fast5.tar ; mv UBC/* data/fast5/ ; rm FAB42473-4179682758_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42395-4178605061_Multi_Fast5.tar
tar -xvf FAB42395-4178605061_Multi_Fast5.tar ; mv Norwich/* data/fast5/ ; rm FAB42395-4178605061_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42476-3843483077_Multi_Fast5.tar
tar -xvf FAB42476-3843483077_Multi_Fast5.tar ; mv UBC/* data/fast5/ ; rm FAB42476-3843483077_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42451-4239353418_Multi_Fast5.tar
tar -xvf FAB42451-4239353418_Multi_Fast5.tar ; mv Notts/* data/fast5/ ; rm FAB42451-4239353418_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42704-87746129_Multi_Fast5.tar
tar -xvf FAB42704-87746129_Multi_Fast5.tar ; mv UBC/* data/fast5/ ; rm FAB42704-87746129_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42828-288548394_Multi_Fast5.tar
tar -xvf FAB42828-288548394_Multi_Fast5.tar ; mv Norwich/* data/fast5/ ; rm FAB42828-288548394_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42810-352384898_Multi_Fast5.tar
tar -xvf FAB42810-352384898_Multi_Fast5.tar ; mv Norwich/* data/fast5/ ; rm FAB42810-352384898_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42798-3306352129_Multi_Fast5.tar
tar -xvf FAB42798-3306352129_Multi_Fast5.tar ; mv Norwich/* data/fast5/ ; rm FAB42798-3306352129_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB45280-222619780_Multi_Fast5.tar
tar -xvf FAB45280-222619780_Multi_Fast5.tar ; mv Norwich/* data/fast5/ ; rm FAB45280-222619780_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB46664-288286712_Multi_Fast5.tar
tar -xvf FAB46664-288286712_Multi_Fast5.tar ; mv UBC/* data/fast5/ ; rm FAB46664-288286712_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB46683-4246923067_Multi_Fast5.tar
tar -xvf FAB46683-4246923067_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAB46683-4246923067_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB45332-551111640_Multi_Fast5.tar
tar -xvf FAB45332-551111640_Multi_Fast5.tar ; mv UBC/* data/fast5/ ; rm FAB45332-551111640_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB43577-3574887596_Multi_Fast5.tar
tar -xvf FAB43577-3574887596_Multi_Fast5.tar ; mv UCSC/* data/fast5/ ; rm FAB43577-3574887596_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB44989-2567311907_Multi_Fast5.tar
tar -xvf FAB44989-2567311907_Multi_Fast5.tar ; mv UCSC/* data/fast5/ ; rm FAB44989-2567311907_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF01169-4245879798_Multi_Fast5.tar
tar -xvf FAF01169-4245879798_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF01169-4245879798_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF01441-3910073345_Multi_Fast5.tar
tar -xvf FAF01441-3910073345_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF01441-3910073345_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB45277-86567043_Multi_Fast5.tar
tar -xvf FAB45277-86567043_Multi_Fast5.tar ; mv Notts/* data/fast5/ ; rm FAB45277-86567043_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB45321-19064779_Multi_Fast5.tar
tar -xvf FAB45321-19064779_Multi_Fast5.tar ; mv Notts/* data/fast5/ ; rm FAB45321-19064779_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF01127-353303576_Multi_Fast5.tar
tar -xvf FAF01127-353303576_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF01127-353303576_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF01132-84868110_Multi_Fast5.tar
tar -xvf FAF01132-84868110_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF01132-84868110_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB49712-622291475_Multi_Fast5.tar
tar -xvf FAB49712-622291475_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAB49712-622291475_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF01253-83756522_Multi_Fast5.tar
tar -xvf FAF01253-83756522_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF01253-83756522_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB45321-285174896_Multi_Fast5.tar
tar -xvf FAB45321-285174896_Multi_Fast5.tar ; mv Notts /* data/fast5/ ; rm FAB45321-285174896_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB49914-3775529215_Multi_Fast5.tar
tar -xvf FAB49914-3775529215_Multi_Fast5.tar ; mv Notts /* data/fast5/ ; rm FAB49914-3775529215_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB45271-152889212_Multi_Fast5.tar
tar -xvf FAB45271-152889212_Multi_Fast5.tar ; mv Notts /* data/fast5/ ; rm FAB45271-152889212_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB49164-4045668814_Multi_Fast5.tar
tar -xvf FAB49164-4045668814_Multi_Fast5.tar ; mv UCSC/* data/fast5/ ; rm FAB49164-4045668814_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB49908-481119249_Multi_Fast5.tar
tar -xvf FAB49908-481119249_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAB49908-481119249_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF04090-3842965088_Multi_Fast5.tar
tar -xvf FAF04090-3842965088_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF04090-3842965088_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF15665-16056159_Multi_Fast5.tar
tar -xvf FAF15665-16056159_Multi_Fast5.tar ; mv Notts/* data/fast5/ ; rm FAF15665-16056159_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF13748-17958431_Multi_Fast5.tar
tar -xvf FAF13748-17958431_Multi_Fast5.tar ; mv Notts/* data/fast5/ ; rm FAF13748-17958431_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF10039-2901545329_Multi_Fast5.tar
tar -xvf FAF10039-2901545329_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF10039-2901545329_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF09968-3439856925_Multi_Fast5.tar
tar -xvf FAF09968-3439856925_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF09968-3439856925_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF09277-3709819546_Multi_Fast5.tar
tar -xvf FAF09277-3709819546_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF09277-3709819546_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF14035-3976726082_Multi_Fast5.tar
tar -xvf FAF14035-3976726082_Multi_Fast5.tar ; mv Notts/* data/fast5/ ; rm FAF14035-3976726082_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF15694-4109802543_Multi_Fast5.tar
tar -xvf FAF15694-4109802543_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF15694-4109802543_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF09713-4111860526_Multi_Fast5.tar
tar -xvf FAF09713-4111860526_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF09713-4111860526_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF18554-4178920553_Multi_Fast5.tar
tar -xvf FAF18554-4178920553_Multi_Fast5.tar ; mv UBC/* data/fast5/ ; rm FAF18554-4178920553_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF15630-4244782843_Multi_Fast5.tar
tar -xvf FAF15630-4244782843_Multi_Fast5.tar ; mv Notts/* data/fast5/ ; rm FAF15630-4244782843_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF09640-4245291640_Multi_Fast5.tar
tar -xvf FAF09640-4245291640_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF09640-4245291640_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF09701-4249180049_Multi_Fast5.tar
tar -xvf FAF09701-4249180049_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF09701-4249180049_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF15586-82266371_Multi_Fast5.tar
tar -xvf FAF15586-82266371_Multi_Fast5.tar ; mv Bham/* data/fast5/ ; rm FAF15586-82266371_Multi_Fast5.tar
wget http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF05869-87644245_Multi_Fast5.tar
tar -xvf FAF05869-87644245_Multi_Fast5.tar ; mv UBC/* data/fast5/ ; rm FAF05869-87644245_Multi_Fast5.tar


#1) Create or change directory where you want to reproduce the results.

#2) Download and extract the data
#   First, we download the PacBio fast5 file from: 
#   https://trace.ncbi.nlm.nih.gov/Traces/?view=run_browser&acc=SRR15058166&display=metadata
#   Note: this may be slow as the FAST5 files total over 4TB
#   We also clean up the directory since these files are very large

Download the data
mkdir data/fast5/



#3) Clone the github repository
git clone https://github.com/bayesomicslab/ONT-nonb-GoFAE-DND.git
```

## Reads Processing:


<p align="center">

  <img width=60% height=60% src="https://user-images.githubusercontent.com/45966768/228649149-8bdf2bf2-043e-4a58-bc6a-ec0264b991b5.png">

</p>



Sequence bases are called from the raw ONT current using [Albacore](http://porecamp.github.io/2017/basecalling.html),
which generates an event table that describes the DNA context in the
nanopore (Loman et al., [2015](https://www.nature.com/articles/nmeth.3444)). 

#### [Albacore Basecalling](http://porecamp.github.io/2017/basecalling.html):

```Albacore
~$ read_fast5_basecaller.py -f FLO-PRO002 -k SQK-LSK109 --input $path/na12878/fast5/single/ --save_path $path/na12878/fast5/albacore_single/ --output_format fastq,fast5 -t 48 --recursive --config r941_450bps_linear_prom.cfg
```


Subsequently, we [re-squiggle](https://nanoporetech.github.io/tombo/resquiggle.html) the FAST5  output of Albacore using [Tombo](https://nanoporetech.github.io/tombo/tutorials.html), 
a statistical method that detects base modifications in nanopore current signal (Stoiber et al., [2017](https://www.biorxiv.org/content/10.1101/094672v2.abstract)). 
Briefly, the re-squiggling algorithm segments the raw current signal into events and calls nucleotide bases using the current and a reference genome for
correcting spurious variation ([Figures, top](https://user-images.githubusercontent.com/45966768/228670741-a137b6bc-c7af-464e-a50e-489cc1f1fb19.PNG) and [middle](https://user-images.githubusercontent.com/45966768/228649149-8bdf2bf2-043e-4a58-bc6a-ec0264b991b5.png)).

The Tombo segmentation provides current measurements at the base-level, unlike Albacore, which assumes the block stride
attribute remains fixed, which enables the computation of translocation times ([Figures, bottom](https://user-images.githubusercontent.com/45966768/228670741-a137b6bc-c7af-464e-a50e-489cc1f1fb19.PNG) and [bottom](https://user-images.githubusercontent.com/45966768/228649149-8bdf2bf2-043e-4a58-bc6a-ec0264b991b5.png)).
For each position on the Tombo-mapped reads, we compute the time duration in seconds as the ratio of the number of current measurements to the ONT sampling rate.

<p align="center">

  <img width=90% height=90% src="https://user-images.githubusercontent.com/45966768/228670741-a137b6bc-c7af-464e-a50e-489cc1f1fb19.PNG">

</p>

#### [Tombo re-squiggle](https://nanoporetech.github.io/tombo/resquiggle.html):

```Tombo
~$ tombo resquiggle $path/workspace/pass/ hg38.fa --dna --overwrite --basecall-group Basecall_1D_001 --include-event-stdev --failed-reads-filename $path/workspace/pass/tombo_failed_reads.txt --processes 48
```




## Prepare the windows:

### Step 1:
Extract motifs postions from [non-B DNA DB](https://nonb-abcc.ncifcrf.gov/apps/site/default).

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


## Simulation


This is an example of command that simulate
$100,000$ B-DNA and $1,000$ non-B DNA windows 
for G-quadruples and Short Tandem Repeat.

```simulation
~$ simulator.py -nb 1000 -b 100000 
```




## GoFAE-DND method:

<p align="center">

  <img width=60% height=60% src="https://user-images.githubusercontent.com/45966768/228676731-1c8ac6a9-8221-42db-aedd-d4e8096f9331.png">

</p>
