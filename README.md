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

## Cloning the Project and Creating the Environment

These instructions assume that you have miniconda or anaconda installed.
To begin, clone the GoFAE-DND project. In a command line with git in your path:
```
$ git clone https://github.com/bayesomicslab/ONT-nonb-GoFAE-DND.git
```
Then, change directory into the ONT-nonb-GoFAE-DND/ folder and
create and activate the conda environment.

```
$ conda env create -f GoFAE-DND/gofaednd/gofaednd_env.yml
$ conda activate gofaednd_env
```

## Simulations

#### 1. Make simulated windows

This is an example of command that simulate
100,000 B-DNA and 1,000 non-B DNA windows 
for G-quadruples and Short Tandem Repeat.
You must have matplotlib, numpy, and pandas 
installed.
In a command line with python3 in your path, 
change directory into simulator/ and then
run the simulator, which specifies the amount 
of non-B and B DNA samples. 
This may take a couple minutes:

     $ cd simulator
     $ python3 simulator.py -nb 10000 -b 1000000 

#### 2. Run GoFAE-DND on simulated data
All python dependencies for the GoFAE-DND are in [`gofaednd_env.yml`](GoFAE-DND/gofaednd/gofaednd_env.yml)

```
$ cd ../GoFAE-DND/gofaednd
$ python3 Main.py --data_type=simulated --sim_data_path=../../simulated_data/ --config=1 --nonb_type=G_Quadruplex_Motif --nonb_ratio=0.1 --n_z=64 --num_projections=64 --epochs=25 --fdr_level=0.2 --discriminative_weight=35 --lambda_alpha=0.5 
$ python3 Main.py --data_type=simulated --sim_data_path=../../simulated_data/ --config=2 --nonb_type=Short_tandem_repeat --nonb_ratio=0.1 --n_z=64 --num_projections=64 --epochs=25 --fdr_level=0.2 --discriminative_weight=35 --lambda_alpha=0.5 
```

Note: to get a list of all parameters, run the command:
```
$ python3 Main.py --help
```

The results including spreadsheets of metrics and diagnostic images 
are included in the root-level results/ directory.

#### 3. Run novelty detection methods

To run the other methods, we will need to install seaborn and h5py first.

```
$ conda install seaborn
$ conda install h5py
```
Afterwards, we can run isolation forests, local outlier factors, and one class SVMs.

  ```
  $ cd ../../novelty_detectors
  $ python3 isolation_forest.py -W ignore -d sim -f ../simulated_data/ -r ../results/ -nb 20000 -b 200000 
  $ python3 local_outlier_factor.py -W ignore -d sim -f ../simulated_data/ -r ../results/ -nb 20000 -b 200000 
  $ python3 svm_one_class.py -W ignore -d sim -f ../simulated_data/ -r ../results/ -nb 20000 -b 200000   
  ```



#### 4. Run classifiers

```
$ cd ../classifiers
$ python3 -W ignore svc.py -d sim -f ../simulated_data/ -r ../results/ -nb 20000 -b 200000
$ python3 -W ignore random_forest.py -d sim -f ../simulated_data/ -r ../results/ -nb 20000 -b 200000
$ python3 -W ignore nearest_neighbors.py -d sim -f ../simulated_data/ -r ../results/ -nb 20000 -b 200000
$ python3 -W ignore logistic_regression.py -d sim -f ../simulated_data/ -r ../results/ -nb 20000 -b 200000
$ python3 -W ignore gaussian_process.py -d sim -f ../simulated_data/ -r ../results/ -nb 20000 -b 200000
```

The results for both novelty detectors and classifiers can be found in the root-level
results directory.

[//]: # ()
[//]: # (#### 5. Compare the methods: &#40;Reproduce Fig 6 and S4&#41;:)

[//]: # ()
[//]: # (```)

[//]: # ($ cd ../)

[//]: # ($ python3 plottings.py)

[//]: # (```)


[//]: # (## GoFAE-DND method:)

<p align="center">

  <img width=60% height=60% src="https://user-images.githubusercontent.com/45966768/228676731-1c8ac6a9-8221-42db-aedd-d4e8096f9331.png">

</p>





## Experimental data

### Instructions for downloading the data and GitHub repo

   1) Create or change directory where you want to reproduce the results.

   2) Download and extract the data.

       First, we download the PacBio fast5 file from: 
       https://trace.ncbi.nlm.nih.gov/Traces/?view=run_browser&acc=SRR15058166&display=metadata
       ### Note: this may be slow as the FAST5 files total over 4TB 
       For convenience, we provide a download.sh script in the script folder that will 
       download, unpack, and clean up the directory.
        
        ```
        $ sh download.sh
        ```


### Processing Reads

<p align="center">

  <img width=60% height=60% src="https://user-images.githubusercontent.com/45966768/228649149-8bdf2bf2-043e-4a58-bc6a-ec0264b991b5.png">

</p>



Sequence bases are called from the raw ONT current using [Albacore](http://porecamp.github.io/2017/basecalling.html),
which generates an event table that describes the DNA context in the
nanopore (Loman et al., [2015](https://www.nature.com/articles/nmeth.3444)). 

#### [Albacore Basecalling](http://porecamp.github.io/2017/basecalling.html):

```Albacore
 $ read_fast5_basecaller.py -f FLO-PRO002 -k SQK-LSK109 --input $path/na12878/fast5/single/ --save_path $path/na12878/fast5/albacore_single/ --output_format fastq,fast5 -t 48 --recursive --config r941_450bps_linear_prom.cfg
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
$ tombo resquiggle $path/workspace/pass/ hg38.fa --dna --overwrite --basecall-group Basecall_1D_001 --include-event-stdev --failed-reads-filename $path/workspace/pass/tombo_failed_reads.txt --processes 48
```




### Prepare the windows

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




