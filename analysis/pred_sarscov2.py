import pickle
import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import math
import sys

from keras.models import Model, load_model
from keras.layers import (
    Input, Dense, Embedding, Conv1D, Flatten, Concatenate,
    MaxPooling1D, Dropout, Dot, LeakyReLU
)
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from keras.utils import multi_gpu_model, Sequence, np_utils
import scipy.stats as ss

MAXLEN = 1000
epochs = 10
num_gpus = 1
batch_size = 1500*num_gpus

thres = float(sys.argv[1])
option = sys.argv[2]
tid = sys.argv[3]
embedding_file = f"data/julia_embed_{option}_corona.txt"
print("option: ", option, "threshold: ", thres)

model_file = f'models/model_joint_corona_{option}_{thres}_{tid}.h5'
out_file = f'stats/tune_joint_corona_{option}_{thres}_{tid}.pckl'
preds_file = f'preds/preds_joint_corona_{option}_{thres}_{tid}.txt'
open_preds = open(preds_file, "w")
open_preds.close()

swissprot_file = '../data/swissprot-proteome.tab'
hpi_file = '../data/hpi.virus.family.txt'
corona_interactions = 'data/media-6.xlsx'
corona_sequences = 'data/2020-04-krogan-sarscov2-sequences-uniprot-mapping.xlsx'

pi = 11
params = {}
if pi != -1:
    max_kernels = [17, 33, 65]
    nb_filters = [8, 16]
    dense_units = [8, 16, 32]
    pool_sizes = [50, 200]
    params['max_kernel'] = max_kernels[pi % len(max_kernels)]
    pi //= len(max_kernels)
    params['nb_filters'] = nb_filters[pi % len(nb_filters)]
    pi //= len(nb_filters)
    params['pool_size'] = pool_sizes[pi % len(pool_sizes)]
    pi //= len(pool_sizes)
    params['dense_units'] = dense_units[pi % len(dense_units)]
    pi //= len(dense_units)
print('Params:', params)

def repeat_to_length(s, length):
    return (s * (length//len(s) + 1))[:length]

def to_onehot(seq, option, start=0):
    onehot = np.zeros((MAXLEN, 22), dtype=np.int32)
    seq = repeat_to_length(seq, MAXLEN)
    l = min(MAXLEN, len(seq))
    if l != 1000:
        print("Wrong")
    if option == 'virus':
        for i in range(start, start + l):
            onehot[i, vaaindex.get(seq[i - start], 0)] = 1
    elif option == 'human':
        for i in range(start, start + l):
            onehot[i, haaindex.get(seq[i - start], 0)] = 1
    onehot[0:start, 0] = 1
    onehot[start + l:, 0] = 1
    return onehot

vaaletter = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','X','Y']
haaletter = ['A','H','Y','F','P','M','U','V','C','G','R','W','N','S','T','K','D','L','E','Q','I']
vaaindex = dict()
haaindex = dict()
for i in range(len(vaaletter)):
    vaaindex[vaaletter[i]] = i + 1
for i in range(len(haaletter)):
    haaindex[haaletter[i]] = i + 1

data = pd.read_csv(embedding_file, header = None, sep = ' ', skiprows=1)
embds_data = data.values
embed_dict = dict(zip(embds_data[:,0],embds_data[:,1:-1]))
dim_input = 100
print('finished reading embeddings')

hp_set = set()
prot2embed = {}
with open(swissprot_file, 'r') as f:
    next(f)
    for line in f:
        items = line.strip().split('\t')
        if len(items[3]) > MAXLEN or items[0] not in embed_dict:
            continue
        hp_set.add(items[0])
        prot2embed[items[0]] = to_onehot(items[3], 'human')
print('Number of host proteins: ', len(hp_set))

positives = set()
family_dict = {}
pathogens = set()
family2vp = {}
vp2patho = {}
with open(hpi_file, 'r') as f:
    for line in f:
        items = line.strip().split('\t')
        if 'family' not in items[4] or 'miscore:' not in items[5] \
                    or len(items[6]) > MAXLEN or len(items[7]) > MAXLEN or items[0] not in hp_set:
            continue
        if float(items[5].split('miscore:')[1]) >= thres:
            hp = items[0]
            vp = items[1]
            patho = '<http://purl.obolibrary.org/obo/NCBITaxon_' + items[2] + '>'
            if hp not in embed_dict or patho not in embed_dict:
                continue
            family = '<http://purl.obolibrary.org/obo/NCBITaxon_' + items[3] + '>'
            if family == '<http://purl.obolibrary.org/obo/NCBITaxon_11118>':
                continue
            prot2embed[vp] = to_onehot(items[7], 'virus')
            family_dict[patho] = family
            positives.add((hp, vp, patho, family))
            pathogens.add(patho)
            if family not in family2vp:
                family2vp[family] = set()
            family2vp[family].add(vp)
            vp2patho[vp] = patho

dfs_interaction = pd.read_excel(corona_interactions, sheet_name = None, skiprows=1)
for index, row in dfs_interaction["Sheet1"].iterrows():
    hp = row['Preys']
    vp = row['Bait']
    patho = '<http://purl.obolibrary.org/obo/NCBITaxon_2697049>'
    family = '<http://purl.obolibrary.org/obo/NCBITaxon_11118>'
    if hp not in embed_dict or hp not in prot2embed:
            continue
    family_dict[patho] = family
    positives.add((hp, vp, patho, family))
    pathogens.add(patho)
    if family not in family2vp:
        family2vp[family] = set()
    family2vp[family].add(vp)
    vp2patho[vp] = patho
print('Number of positives: ', len(positives))
print('Number of pathogens: ', len(pathogens))
print('Number of families: ', len(family2vp))

dfs_seq = pd.read_excel(corona_sequences)
for index, row in dfs_seq.iterrows():
    vp = row['Krogan name']
    seq = row['Sequence']
    if '*' == seq[len(seq)-1]:
        seq = seq[:len(seq)-1]
    prot2embed[vp] = to_onehot(seq, 'virus')

family_counts = {}
for triple in positives:
    family = family_dict[triple[2]]
    if family not in family_counts:
        family_counts[family] = 0
    family_counts[family] += 1
    
family_thres = int(len(positives)*0.01)
family_thres = 0
filtered_families = set()
for family in family_counts:
    if family_counts[family] >= family_thres:
        filtered_families.add(family)
print("After filtering with threshold %d, there remain %d families" % (family_thres, len(filtered_families)))

vp_set = set()
for family in filtered_families:
    vp_set = vp_set | family2vp[family]
print('Number of virus proteins after filtering: ', len(vp_set))

filtered_pathogens = set()
filtered_positives = set()
for positive in positives:
    if positive[3] in filtered_families:
        filtered_positives.add(positive)
        filtered_pathogens.add(positive[2])
print('Number of positives after filtering: ', len(filtered_positives))
print('Number of pathogens after filtering: ', len(filtered_pathogens))
positives = filtered_positives


class Generator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.nbatch = int(np.ceil(len(self.x) / float(self.batch_size)))
        self.length = len(self.x)

    def __len__(self):
        return self.nbatch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        batch_len = min(self.batch_size, (self.length)-start)
        x_seq1 = np.empty((batch_len, 1000,22), dtype=np.float32)
        x_seq2 = np.empty((batch_len, 1000,22), dtype=np.float32)
        x_pheno1 = np.empty((batch_len, dim_input), dtype=np.float32)
        x_pheno2 = np.empty((batch_len, dim_input), dtype=np.float32)
        y_batch = np.empty(batch_len, dtype=np.float32)

        for ids in range(start, min((idx + 1) * self.batch_size, self.length)):
            x_seq1[ids-start,:,:] = prot2embed[self.x[ids][0]]
            x_seq2[ids-start,:,:] = prot2embed[self.x[ids][1]]
            x_pheno1[ids-start,:] = embed_dict[self.x[ids][0]]
            x_pheno2[ids-start,:] = embed_dict[self.x[ids][2]]
            y_batch[ids-start] = self.y[ids]
        return [x_seq1, x_seq2, x_pheno1, x_pheno2], y_batch

def complete_triple(vp_set, positives_set, triples, option="valtest"):
    for hp in hp_set:
            for vp in vp_set:
                pair = (hp, vp)
                if pair not in positives_set:
                    if option == 'train':
                        triples.append((hp, vp, vp2patho[vp], 0))
                    else:
                        triples[pair[1]].append((hp, vp, vp2patho[vp], 0))
    if option != 'train':
        for key in triples.keys():
            triples[key] = np.array(triples[key])
        
def split_positives(families, option="valtest"):
    positives_set = set()
    if option == 'train':
        triples = []
        for items in positives:
            if items[3] in train_families:
                positives_set.add((items[0], items[1]))
                triples.append((items[0], items[1], items[2], 1))
    else:
        triples = {}
        for items in positives:
            if items[3] in families:
                    positives_set.add((items[0], items[1]))
                    if items[1] not in triples:
                        triples[items[1]] = []
                    triples[items[1]].append((items[0], items[1], items[2], 1))
                    if items[1] not in num_positive:
                        num_positive[items[1]] = 0
                    num_positive[items[1]] += 1
    return triples, positives_set

def get_model():
    seq = Input(shape=(MAXLEN, 22), dtype=np.float32)
    kernels = range(8, params['max_kernel'], 8)
    nets = []
    for i in range(len(kernels)):
        conv = Conv1D(
            filters=params['nb_filters'],
            kernel_size=kernels[i],
            padding='valid',
            kernel_initializer='glorot_normal')(seq)
        print(conv.get_shape())
        conv_dropout = Dropout(0.5)(conv)
        pool = MaxPooling1D(pool_size=params['pool_size'])(conv_dropout)
        flat = Flatten()(pool)
        nets.append(flat)

    net = Concatenate(axis=1)(nets)
    dense_seq = Dense(params['dense_units'])(net)
    activ_seq = LeakyReLU(alpha=0.1)(dense_seq)
    dropout_seq = Dropout(0.5)(activ_seq)

    pheno = Input(shape=(100,))
    dense_pheno = Dense(params['dense_units'])(pheno)
    activ_pheno = LeakyReLU(alpha=0.1)(dense_pheno)
    dropout_pheno = Dropout(0.5)(activ_pheno)

    flat = Concatenate(axis=-1)([dropout_seq, dropout_pheno])
    return seq, pheno, flat

def compute_auc(ranks):
    auc_x, auc_x_counts = np.unique(ranks, return_counts=True)
    auc_y = []
    tpr = 0
    increment = 1/len(ranks)
    for i in range(len(auc_x)):
        tpr += auc_x_counts[i] * increment
        auc_y.append(tpr)
    auc_x = np.append(auc_x, len(hp_set))
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x)/len(hp_set)
    return auc
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)
K.set_session(sess)

counter = 0
fold_ranks = {}
family_aucs = []
for test_family in ["<http://purl.obolibrary.org/obo/NCBITaxon_11118>"]:
    counter+=1
    print('Test family %d: %s' % (counter, test_family))
    K.clear_session()
    tv_families = list(filtered_families - set([test_family]))
    val_families = set(np.random.choice(tv_families, size = math.ceil(len(tv_families)/5), replace=False))
    train_families = set(tv_families) - val_families

    print('Train families: ', len(train_families), 'validation families', len(val_families))

    num_positive = {} 
    train_positives, train_positives_set = split_positives(train_families, option="train")
    triple_val, val_positives_set = split_positives(val_families)
    triple_test, test_positives_set = split_positives(test_family)
    print("Number of positives in train, validation and test families:", len(train_positives), len(val_positives_set), len(test_positives_set))
    print("Number of proteins in validation and test families:", len(triple_val.values()), len(triple_test.values()))
    
    train_negatives = []
    val_vp_set = set()
    for family in val_families:
        val_vp_set = val_vp_set | family2vp[family]
    train_vp_set = vp_set - family2vp[test_family] - val_vp_set
    
    complete_triple(train_vp_set, train_positives_set, train_negatives, option='train')
    complete_triple(val_vp_set, val_positives_set, triple_val)
    complete_triple(family2vp[test_family], test_positives_set, triple_test)

    train_positives = np.repeat(np.array(list(train_positives)), len(train_negatives)//len(train_positives), axis = 0)
    train_negatives = np.array(train_negatives)
    triple_train = np.concatenate((train_positives, train_negatives), axis=0)
    np.random.shuffle(triple_train)
    
    triple_val_matrix = np.empty((len(val_vp_set), len(hp_set), 4), dtype = object)
    vp_ordered = []
    val_vp_set = list(val_vp_set)
    for i in range(len(val_vp_set)):
        triple_val_matrix[i, :, :] = triple_val[val_vp_set[i]]
        vp_ordered.append(val_vp_set[i])

    triple_val_matrix = triple_val_matrix.reshape(triple_val_matrix.shape[0]*triple_val_matrix.shape[1], 4)
    print("triple_val_matrix.shape, len(vp_ordered)", triple_val_matrix.shape, len(vp_ordered))
    
    seq1, pheno1, flat1 = get_model()
    seq2, pheno2, flat2 = get_model()
    
    final = Dot(axes=-1, normalize=True)([flat1, flat2])
    output = Dense(1, activation='sigmoid')(final)

    model = Model(inputs=[seq1, seq2, pheno1, pheno2], outputs=output)
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])
    generator = Generator(triple_train[:,0:3], triple_train[:,3], batch_size)
    
    val_maxauc = 0
    for i in range(epochs):
        print('taxon ', counter, ' epoch ', i)
        history = model.fit_generator(generator=generator,
                            epochs=1,
                            steps_per_epoch = 500,
                            verbose=2,
                            max_queue_size = 50,
                            use_multiprocessing=False,
                            workers = 1)
        epoch_ranks = np.array([])
        sim_list = model.predict_generator(generator=Generator(triple_val_matrix[:,0:3], triple_val_matrix[:,3], batch_size), 
                                                    verbose=2, steps=int(math.ceil(triple_val_matrix.shape[0]/batch_size)), 
                                                    max_queue_size = 30, use_multiprocessing=False, workers = 1)
            
        for i in range(len(vp_ordered)):
            vp = vp_ordered[i]
            y_rank = ss.rankdata(-sim_list[i*len(hp_set):(i+1)*len(hp_set)], method='average')
            x_list = y_rank[:num_positive[vp]]
            epoch_ranks = np.concatenate((epoch_ranks, x_list))
        
        val_auc = compute_auc(epoch_ranks)
        print('The AUC for this epoch is ', val_auc)
        if val_auc > val_maxauc:
            print('Saving current model...')
            model.save(model_file)
            val_maxauc = val_auc

    test_ranks = np.array([])
    del model
    K.clear_session()
    model = load_model(model_file)
    for vp in family2vp[test_family]:
        print('vp: ', vp)
        sim_list = model.predict_generator(generator=Generator(triple_test[vp][:,0:3], triple_test[vp][:,3], batch_size), 
                                                verbose=2, steps=math.ceil(len(triple_test[vp])/batch_size), 
                                                max_queue_size = 30, use_multiprocessing=False, workers = 1)
        y_rank = ss.rankdata(-sim_list, method='average')
        x_list = y_rank[:num_positive[vp]]
        test_ranks = np.concatenate((test_ranks, x_list))
        with open(preds_file, 'a+') as f:
            for i in range(triple_test[vp].shape[0]):
                f.write("%s\t%s\t%s\t%s\t%f\t%s\t%d\n" % (vp, triple_test[vp][i,0], triple_test[vp][i,2], test_family, sim_list[i], i<num_positive[vp], len(hp_set)))

    test_auc = compute_auc(test_ranks)
    print('The AUC for the test family is ', test_auc)
    family_aucs.append(test_auc)
    fold_ranks[test_family] = test_ranks

final_auc = compute_auc(np.concatenate(list(fold_ranks.values())))
print('Micro AUC is %f and Macro AUC is %f' % (final_auc, np.mean(family_aucs)))
with open(out_file, 'wb') as fp:
    pickle.dump((len(hp_set), fold_ranks), fp)
