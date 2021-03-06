# DeepViral: prediction of novel virus–hostinteractions from protein sequences and infectiousdisease phenotypes
This is the repository for the DeepViral paper, with source code for the models and analysis, as well as the datasets required. \
The manuscript is now available on [Biorxiv](http://biorxiv.org/content/10.1101/2020.04.22.055095v2). \
Please contact liuwei.wang@kaust.edu.sa for any questions regarding the code and the manuscript.

### Software environment 
Python 3.6.8   
Keras 2.2.4  
Tensorflow 1.13.1  

### To reproduce DeepViral results
#### Leave-One-Family-Out (LOFO)
```
python3 deepviral.py <option> data/julia_embed_cleaned.txt <tid>
```
Option can be seq/human/viral/joint, corresponding to the four DeepViral variants in Table 1. \
Tid can be an arbitrary integer, e.g. 0-4. The prediction results will be in ```preds_option_tid.txt```.\
After 5 runs, run 
```
python3 deepviral.py preds_joint_
``` 
to obtain the confidence intervals and mean ranks.

#### Leave-One-Species-Out (LOSO)
For Zika/Influenza/HPV, run
```
python3 deepviral_taxon_sars2.py joint data/julia_embed_cleaned.txt <test taxon ID> <val taxon ID> <family taxon ID> <evaluation> <tid>
```
Evaluation can be species or family, i.e. LOSO or LOFO, respectively.

For SARS-CoV-2, run
```
python3 deepviral_taxon_sars2.py joint data/julia_embed_cleaned.txt 2697049 694009 11118 species/family <tid>
```

### Directories
#### ./rf and ./rcnn: 
Implementation of Doc2Vec + RF and RCNN on our dataset for the results in Table 1.\
Similarly to the above, the results can be reproduced by 
```
python3 rf.py/rcnn.py <option> data/julia_embed_cleaned.txt <tid>
```

#### ./compare_denovo: 
The code for running DeepViral and RCNN on the DeNovo dataset for Supplementary Table 1.\
To reproduce the results, run
```
python3 compare_denovo_deepviral/rcnn.py <option> ../data/julia_embed_cleaned.txt
```
For rcnn, no option is needed. For DeepViral, option can be seq/human/viral/joint.
The input datasets of the DeNovo dataset are downloaded from the websites of [DeNovo](https://bioinformatics.cs.vt.edu/~alzahraa/denovo) by [Eid et al. (2016)](https://academic.oup.com/bioinformatics/article/32/8/1144/1744545) and [VirusHostPPI](http://165.246.44.47/VirusHostPPI/Additional) by [Zhou et al. (2018)](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-018-4924-2).

### Datasets
[HPIDB 3.0](https://hpidb.igbb.msstate.edu/): a database of host pathogen interactions\
[PathoPhenoDB](http://patho.phenomebrowser.net/#/downloads): a database of pathogen phenotypes\
[HPO](https://hpo.jax.org/app/download/annotation): phenotype annotations of human genes\
[MGI](http://www.informatics.jax.org/downloads/reports/index.html#pheno): phenotype annotations of mouse genes and orthologous mappings to human genes\
[GO](http://current.geneontology.org/products/pages/downloads.html): function annotations of human proteins

### DL2Vec 
DL2Vec is available at https://github.com/bio-ontology-research-group/DL2Vec\
The input ontologies to DL2Vec are available here: [PhenomeNet](http://aber-owl.net/ontology/PhenomeNET/#/), [NCBI Taxonomy](https://www.ebi.ac.uk/ols/ontologies/ncbitaxon)\
To reproduce the embeddings, the association file is provided in ```data/all_asso.txt```.
