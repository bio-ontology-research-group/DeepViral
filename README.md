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
python3 deepviral_taxon.py joint data/julia_embed_cleaned.txt <test taxon ID> <val taxon ID> <family taxon ID> <evaluation> <tid>
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
If you use DeepViral for your research, please cite our [paper](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btab147/6158034) in Bioinformatics:
```
@article{10.1093/bioinformatics/btab147,
    author = {Liu-Wei, Wang and Kafkas, Şenay and Chen, Jun and Dimonaco, Nicholas J and Tegnér, Jesper and Hoehndorf, Robert},
    title = "{DeepViral: prediction of novel virus–host interactions from protein sequences and infectious disease phenotypes}",
    journal = {Bioinformatics},
    year = {2021},
    month = {03},
    abstract = "{Infectious diseases caused by novel viruses have become a major public health concern. Rapid identification of virus–host interactions can reveal mechanistic insights into infectious diseases and shed light on potential treatments. Current computational prediction methods for novel viruses are based mainly on protein sequences. However, it is not clear to what extent other important features, such as the symptoms caused by the viruses, could contribute to a predictor. Disease phenotypes (i.e., signs and symptoms) are readily accessible from clinical diagnosis and we hypothesize that they may act as a potential proxy and an additional source of information for the underlying molecular interactions between the pathogens and hosts.We developed DeepViral, a deep learning based method that predicts protein–protein interactions (PPI) between humans and viruses. Motivated by the potential utility of infectious disease phenotypes, we first embedded human proteins and viruses in a shared space using their associated phenotypes and functions, supported by formalized background knowledge from biomedical ontologies. By jointly learning from protein sequences and phenotype features, DeepViral significantly improves over existing sequence-based methods for intra- and inter-species PPI prediction.Code and datasets for reproduction and customization are available at https://github.com/bio-ontology-research-group/DeepViral. Prediction results for 14 virus families are available at https://doi.org/10.5281/zenodo.4429824.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btab147},
    url = {https://doi.org/10.1093/bioinformatics/btab147},
    note = {btab147},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btab147/36450100/btab147.pdf},
}
```

