import pickle
import numpy as np
import pandas as pd
import sklearn as sk
import sys
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score
from utils import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.ensemble import RandomForestClassifier

def read_swissprot(swissprot_file, embed_dict, first=True, MAXLEN = 1000):
    hp_set = set()
    seq_ids = []
    seqs = []
    with open(swissprot_file, 'r') as f:
        next(f)
        for line in f:
            items = line.strip().split('\t')
            if items[0] not in embed_dict:
                continue
            if first == False and len(items[3]) > MAXLEN:
                continue
            hp_set.add(items[0])
            seq_ids.append(items[0])
            seqs.append(items[3])
    print('Number of host proteins: ', len(hp_set))
    return hp_set, seq_ids, seqs

MAXLEN = 1000

thres = '0'
embedding_file = sys.argv[1]

tid = sys.argv[2]
preds_file = f'preds_rf_{tid}.txt'
open_preds = open(preds_file, "w")
open_preds.close()

swissprot_file = '../data/swissprot-proteome.tab'
hpi_file = '../data/train_1000.txt'

embed_dict = read_embedding(embedding_file)
hp_set, seq_ids, seq_list = read_swissprot(swissprot_file, embed_dict, first = False)

def get_embed(prot2embed, seq_ids, seq_list):
    k, extract_method, vector_size, window, epoch = (5, 2, 32, 3, 70)
    documents = []
    for seq, seq_id in zip(seq_list, seq_ids):
            codes = seq[0: 5000]
            words = [codes[j: j + k] for i in range(k) for j in range(i, len(codes) - (k - 1), k)]
            documents.append(TaggedDocument(words, tags=[seq_id]))

    model = Doc2Vec.load('doc2vec_model/human_virus_all-doc2vector-all-5-2-32-3-70_0-5000_HVPPI_model.pkl')
    protein_encodings = pickle.load(open("doc2vec_model/human_virus_all-doc2vector-all-5-2-32-3-70_0-5000_HVPPI.pkl", 'rb'))

    infernum=0
    unifernum=0
    for seq_id, document in zip(seq_ids, documents):
        # if seq_id in protein_encodings protein_encodings[seq_id] else inferring by doc2vec model
        if seq_id in protein_encodings:
            prot2embed[seq_id] = protein_encodings[seq_id]
            unifernum+=1
        else:
            #print(seq_id, 'is inferred!!!')
            infernum+=1
            prot2embed[seq_id] =list(model.infer_vector(document[0]))
    print('Infered protein number:',infernum)
    print('Uninfered protein number:',unifernum)
    
positives = set()
family_dict = {}
pathogens = set()
family2vp = {}
vp2patho = {}
vp2numPos = {}

with open(hpi_file, 'r') as f:
    next(f)
    for line in f:
        items = line.strip().split('\t')
        if items[0] not in hp_set:
            continue
        if float(items[6]) >= float(thres):
            hp = items[0]
            vp = items[1]
            patho = '<http://purl.obolibrary.org/obo/NCBITaxon_' + items[2] + '>'
            if hp not in embed_dict or patho not in embed_dict:
                continue
            if len(items[5]) > MAXLEN:
                continue
            family = '<http://purl.obolibrary.org/obo/NCBITaxon_' + items[3] + '>'
            seq_ids.append(vp)
            seq_list.append(items[5])
            family_dict[patho] = family
            positives.add((hp, vp, patho, family))
            pathogens.add(patho)
            if family not in family2vp:
                family2vp[family] = set()
            family2vp[family].add(vp)
            vp2patho[vp] = patho
            if vp not in vp2numPos:
                vp2numPos[vp] = 0
            vp2numPos[vp] += 1

vp_set = set(vp2patho.keys())
families = set(family2vp.keys())
print('Number of positives: ', len(positives))
print('Number of pathogens: ', len(pathogens))
print('Number of families: ', len(families))
print('Number of viral proteins: ', len(vp_set))

prot2embed = {}
get_embed(prot2embed, seq_ids, seq_list)
print("len(prot2embed)", len(prot2embed))

def get_triple(positives, families, hp_set, vp_set, vp2patho, option):
    positives_set = set()
    triple_pos = []
    for items in positives:
        if items[3] in families:
            positives_set.add((items[0], items[1]))
            triple_pos.append((items[0], items[1], items[2], 1))
    numPos = len(positives_set)
    print("Number of positives in %s families: %d" % (option, numPos))
    
    triple_neg = []
    for hp in hp_set:
        for vp in vp_set:
            pair = (hp, vp)
            if pair not in positives_set:
                triple_neg.append((hp, vp, vp2patho[vp], 0))
                    
    if option == 'train':
        np.random.shuffle(triple_neg)
        triple_neg = triple_neg[:10*len(triple_pos)]
        triples = np.concatenate((triple_pos, np.array(triple_neg)), axis=0)
        return triples
    else: 
        triples = np.concatenate((np.array(triple_pos), np.array(triple_neg)), axis=0)
    return triples, numPos

counter = 0
family_aucs = []
for test_family in families:
    counter+=1
    print('Test family %d: %s' % (counter, test_family))
    train_families = list(families - set([test_family]))
    print('Train families: ', len(train_families))

    train_vps = set()
    for family in train_families:
        train_vps = train_vps | family2vp[family]
    print("Number of viral proteins in train, val and test: ", len(train_vps), len(family2vp[test_family]))

    triple_train = get_triple(positives, train_families, hp_set, train_vps, vp2patho, 'train')
    triple_test, numPos_test = get_triple(positives, [test_family], hp_set, family2vp[test_family], vp2patho, 'test')
    print("Number of triples in train, val, test", len(triple_train), len(triple_test))
    print(len(triple_train), len(triple_test))
    
    x = []
    y = []
    for triple in triple_train:
        embed1 = prot2embed[triple[0]]
        embed2 = prot2embed[triple[1]]
        x.append(np.concatenate((embed1, embed2)))
        y.append(triple[3])
    print(len(x), len(y))

    rf = RandomForestClassifier(bootstrap=True, class_weight={'1': 10.0, '0': 1.0},
                criterion='entropy', max_depth=None, max_features='auto',
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=1500, n_jobs=20, oob_score=False,
                random_state=None, verbose=0, warm_start=False)
    rf.fit(x, y)

    x_test = []
    y_test = []
    for triple in triple_test:
        embed1 = prot2embed[triple[0]]
        embed2 = prot2embed[triple[1]]
        x_test.append(np.concatenate((embed1, embed2)))
        y_test.append(int(triple[3]))
    print(len(x_test), len(y_test))

    y_score = rf.predict_proba(x_test)
    test_auc = roc_auc_score(y_test, y_score[:,1])
    print(test_auc)
    family_aucs.append(test_auc)
    with open(preds_file, 'a+') as f:
        for i in range(triple_test.shape[0]):
            f.write("%s\t%s\t%s\t%s\t%f\t%s\n" % (triple_test[i,1], triple_test[i,0], triple_test[i,2], test_family, y_score[i, 1], i<numPos_test))

print("mean AUC is: ", np.mean(family_aucs))
