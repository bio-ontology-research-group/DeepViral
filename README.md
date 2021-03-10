# DeepViral: prediction of novel virus–host interactions from protein sequences and infectious disease phenotypes

This is the repository for the DeepViral paper, with source code for the models and analysis, as well as the datasets required.
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

## Cite
If you use DeepViral for your research, please cite our [paper](https://academic.oup.com/bioinformatics/article/36/2/422/5539866) in Bioinformatics:
```
@article{10.1093/bioinformatics/btz595,
    author = {Kulmanov, Maxat and Hoehndorf, Robert},
    title = "{DeepGOPlus: improved protein function prediction from sequence}",
    journal = {Bioinformatics},
    volume = {36},
    number = {2},
    pages = {422-429},
    year = {2019},
    month = {07},
    abstract = {Protein function prediction is one of the major tasks of bioinformatics that can help in wide range of biological problems such as understanding disease mechanisms or finding drug targets. Many methods are available for predicting protein functions from sequence based features, protein–protein interaction networks, protein structure or literature. However, other than sequence, most of the features are difficult to obtain or not available for many proteins thereby limiting their scope. Furthermore, the performance of sequence-based function prediction methods is often lower than methods that incorporate multiple features and predicting protein functions may require a lot of time.We developed a novel method for predicting protein functions from sequence alone which combines deep convolutional neural network (CNN) model with sequence similarity based predictions. Our CNN model scans the sequence for motifs which are predictive for protein functions and combines this with functions of similar proteins (if available). We evaluate the performance of DeepGOPlus using the CAFA3 evaluation measures and achieve an Fmax of 0.390, 0.557 and 0.614 for BPO, MFO and CCO evaluations, respectively. These results would have made DeepGOPlus one of the three best predictors in CCO and the second best performing method in the BPO and MFO evaluations. We also compare DeepGOPlus with state-of-the-art methods such as DeepText2GO and GOLabeler on another dataset. DeepGOPlus can annotate around 40 protein sequences per second on common hardware, thereby making fast and accurate function predictions available for a wide range of proteins.http://deepgoplus.bio2vec.net/.Supplementary data are available at Bioinformatics online.},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btz595},
    url = {https://doi.org/10.1093/bioinformatics/btz595},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/36/2/422/31962785/btz595.pdf},
}

```

