# Prediction of novel virus–host interactions by integrating clinical symptoms and protein sequences
This is the repository for the DeepViral paper, with source code for the models and analysis, as well as the datasets required.

### Software environment 
Python 3.6.8   
Keras 2.2.4  
Tensorflow 1.13.1  

### Directories
./models: the code for running the models for Table 2 of the paper.    
| Models        | Viruses           | Human   |
| ------------- |:-------------:| -----:|
| pheno.py      | Phenotypes | Phenotypes/GO |
| half_pheno.py      | Phenotypes      |   Phenotypes/GO + Sequences |
| seq.py | Sequences |   Sequences |
| half_seq.py | Sequences |   Phenotypes/GO + Sequences |
| joint.py | Phenotypes + Sequences |   Phenotypes/GO + Sequences |

./compare: the code for running the models for Table 1 of the paper
| Models        | Viruses           | Human   |
| ------------- |:-------------:| -----:|
| seq_compare.py      | Sequences | Sequences |
| half_compare.py      | Sequences + Phenotypes  |Sequences |
| joint_compare.py | Phenotypes + Sequences |   Phenotypes/GO + Sequences |

./analysis: the data and code to generate figure 3 and 4
| Results        | Scripts           | Inputs   |
| ------------- |:-------------:| -----:|
| Figure 3      | plot.R | hiv.csv, hepac.csv |
| Figure 4     | plot.R  | familywise.txt |
| Model for SARS-CoV-2 prediction | pred_sarscov2.py |   Phenotypes/GO + Sequences |

### Datasets
[HPIDB 3.0](https://hpidb.igbb.msstate.edu/): a database of host pathogen interactions\
[PathoPhenoDB](http://patho.phenomebrowser.net/#/downloads): a database of pathogen phenotypes\
[HPO](https://hpo.jax.org/app/download/annotation): phenotype annotations of human genes\
[MGI](http://www.informatics.jax.org/downloads/reports/index.html#pheno): phenotype annotations of mouse genes and orthologous mappings to human genes\
[GO](http://current.geneontology.org/products/pages/downloads.html): function annotations of human proteins
