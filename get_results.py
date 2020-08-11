from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.stats import rankdata
import sys

def compute_auc(groups, true, false, rank = False):
    aucs = []
    y_ranks = []
    for group in groups:
        y_pred = np.concatenate((true[group], false[group]))
        y_true = np.concatenate((np.ones(len(true[group])), np.zeros(len(false[group]))))
        auc = roc_auc_score(y_true, y_pred)
        aucs.append(auc)
        if rank == True:
            y_ranks = np.concatenate((y_ranks, rankdata(-y_pred)[:len(true[group])]))
    if rank == True:
        return np.mean(aucs), y_ranks
    return np.mean(aucs)

path = sys.argv[1]
families, taxa, vps = set(), set(), set()
with open('data/train_1000.txt', 'r') as f:
    next(f)
    for line in f:
        items = line.strip().split()
        family = '<http://purl.obolibrary.org/obo/NCBITaxon_' + items[3] + '>'
        taxon = '<http://purl.obolibrary.org/obo/NCBITaxon_' + items[2] + '>'
        vp = items[1]
        families.add(family)
        vps.add(vp)
        taxa.add(taxon)
print(len(families), len(taxa), len(vps))

family_aucss = []
taxon_aucss = []
vp_aucss = []
mean_ranks = []

for i in range(5):
    family_true, family_false, taxon_true, taxon_false, vp_true, vp_false = {}, {}, {}, {}, {}, {}
    for family in families:
        family_true[family] = []
        family_false[family] = []
    print(len(family_false), len(family_true))
    for taxon in taxa:
        taxon_true[taxon] = []
        taxon_false[taxon] = []
    print(len(taxon_true), len(taxon_false))
    for vp in vps:
        vp_true[vp] = []
        vp_false[vp] = []
    print(len(vp_true), len(vp_false))

    with open(f"{path}{i}.txt", 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            vp = items[0]
            taxon = items[2]
            family = items[3]
            label = items[5]
            score = float(items[4])
            if label == "True":
                family_true[family].append(score)
                taxon_true[taxon].append(score)
                vp_true[vp].append(score)
            elif label == "False":
                family_false[family].append(score)
                taxon_false[taxon].append(score)
                vp_false[vp].append(score)

    family_aucss.append(compute_auc(families, family_true, family_false))
    taxon_aucss.append(compute_auc(taxa, taxon_true, taxon_false))
    vp_auc, y_ranks = compute_auc(vps, vp_true, vp_false, rank=True)
    vp_aucss.append(vp_auc)
    mean_ranks.append(np.mean(y_ranks))
#     hit10s = []
#     hit100s = []
#     num_true = 0
#         hit10s.append(len(np.where(y_ranks<=10)[0]))
#         hit100s.append(len(np.where(y_ranks<=100)[0]))
#         num_true += len(vp_true[vp])
#     hit10ss.append(np.sum(hit10s)/num_true)
#     hit100ss.append(np.sum(hit100s)/num_true)
    
def conf_interval(data):
    mean = np.mean(data)
    var = np.var(data)
    interval = 1.96 * np.sqrt(var/len(data))
    print(mean, mean-interval, mean+interval)

conf_interval(vp_aucss)
conf_interval(taxon_aucss)
conf_interval(family_aucss)
print(np.mean(mean_ranks))
