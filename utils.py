from Bio import SeqIO
import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import math

from keras.models import Model, load_model
from keras.layers import (
    Input, Dense, Embedding, Conv1D, Flatten, Concatenate,
    MaxPooling1D, Dropout, Dot, LeakyReLU
)
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from keras.utils import multi_gpu_model, Sequence, np_utils

def get_aaindex(swissprot_file, hpi_file):
    haaletter = set()
    with open(swissprot_file, 'r') as f:
        next(f)
        for line in f:
            items = line.strip().split('\t')
            haaletter = haaletter | set(items[3])
    haaindex = dict()
    haaletter = list(haaletter)
    for i in range(len(haaletter)):
        haaindex[haaletter[i]] = i + 1
        
    vaaletter = set()
    with open(hpi_file, 'r') as f:
        next(f)
        for line in f:
            items = line.strip().split('\t')
            vaaletter = vaaletter | set(items[5])
    vaaindex = dict()
    vaaletter = list(vaaletter)
    for i in range(len(vaaletter)):
        vaaindex[vaaletter[i]] = i + 1
    return haaindex, vaaindex

def get_params():
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
    return params

def repeat_to_length(s, length):
    return (s * (length//len(s) + 1))[:length]

def to_onehot(seq, aaindex, MAXLEN = 1000, repeat=True):
    onehot = np.zeros((MAXLEN, 22), dtype=np.int32)
    original_len = min(MAXLEN, len(seq))
    if repeat == True:
        seq = repeat_to_length(seq, MAXLEN)
    for i in range(MAXLEN):
        onehot[i, aaindex.get(seq[i])] = 1
    onehot[original_len:, 0] = 1
    return onehot

def read_fasta(filename, unip2seq, vp2taxon, vp=False):
    seqs = SeqIO.parse(filename,'fasta')
    for fasta in seqs:
        name, sequence = fasta.id, str(fasta.seq)
        unip = name.split("|")[1]
        unip2seq[unip] = sequence
        if vp == True:
            vp2taxon[unip] = fasta.description.split('OX=')[1].split(' ')[0]
    
def get_index_fasta(file, unip2seq):
    aaletter = set()
    seqs = SeqIO.parse(file,'fasta')
    for fasta in seqs:
        name, sequence = fasta.id, str(fasta.seq)
        unip = name.split("|")[1]
        unip2seq[unip] = sequence
        for aa in set(sequence):
            aaletter.add(aa)
    aaindex = dict()
    aaletter = list(aaletter)
    for i in range(len(aaletter)):
        aaindex[aaletter[i]] = i + 1
    return aaindex

def read_embedding(embedding_file):
    data = pd.read_csv(embedding_file, header = None, sep = ' ', skiprows=1)
    embds_data = data.values
    embed_dict = dict(zip(embds_data[:,0],embds_data[:,1:-1]))
    print('finished reading embeddings')
    return embed_dict

def read_swissprot(swissprot_file, embed_dict, haaindex, repeat = True, first=True, MAXLEN = 1000):
    hp_set = set()
    prot2embed = {}
    with open(swissprot_file, 'r') as f:
        next(f)
        for line in f:
            items = line.strip().split('\t')
            if items[0] not in embed_dict:
                continue
            if first == False and len(items[3]) > MAXLEN:
                continue
            hp_set.add(items[0])
            prot2embed[items[0]] = to_onehot(items[3], haaindex, repeat=repeat)
    print('Number of host proteins: ', len(hp_set))
    return hp_set, prot2embed
        
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
        triple_pos = np.repeat(np.array(triple_pos), len(triple_neg)//len(triple_pos), axis = 0)
        triples = np.concatenate((triple_pos, np.array(triple_neg)), axis=0)
        np.random.shuffle(triples)
        return triples
    if option == 'val':
        np.random.shuffle(triple_neg)
        triples = np.concatenate((np.array(triple_pos), np.array(triple_neg[:int(0.1*len(triple_neg))])), axis=0)
    else: 
        triples = np.concatenate((np.array(triple_pos), np.array(triple_neg)), axis=0)
    return triples, numPos
    






# def get_vtPos(positives, families, triples):
#     positives_set = set()
#     triples = {}
#     for items in positives:
#         if items[3] in families:
#                 positives_set.add((items[0], items[1]))
#                 if items[1] not in triples:
#                     triples[items[1]] = []
#                 triples[items[1]].append((items[0], items[1], items[2], 1))
#     return triples, positives_set

# def split_seq(splited, triples, unip2seq, prot2embed, haaindex, vaaindex, MAXLEN=1000):
#     for pos in triples:
#         hss = [unip2seq[pos[0]]]
#         vss = [unip2seq[pos[1]]]
#         if len(hss[0]) > MAXLEN:
#             hss = [hss[0][i:i+MAXLEN] for i in range(0, len(hss[0])-MAXLEN, 250)]
#         if len(vss[0]) > MAXLEN:
#             vss = [vss[0][i:i+MAXLEN] for i in range(0, len(vss[0])-MAXLEN, 250)]
#         for i in range(len(hss)):
#             for j in range(len(vss)):
#                 hp = pos[0] + '_' + str(i)
#                 vp = pos[1] + '_' + str(j)
#                 prot2embed[hp] = to_onehot(hss[i], haaindex)
#                 prot2embed[vp] = to_onehot(vss[j], vaaindex)
#                 splited.append((hp, vp, pos[2]))
                
# def split_seq_test(splited, triples, unip2seq, prot2embed, haaindex, vaaindex, MAXLEN=1000):
#     counts = []
#     for pos in triples:
#         hss = [unip2seq[pos[0]]]
#         vss = [unip2seq[pos[1]]]
#         if len(hss[0]) > MAXLEN:
#             hss = [hss[0][i:i+MAXLEN] for i in range(0, len(hss[0])-MAXLEN, 250)]
#         if len(vss[0]) > MAXLEN:
#             vss = [vss[0][i:i+MAXLEN] for i in range(0, len(vss[0])-MAXLEN, 250)]
#         count = 0
#         for i in range(len(hss)):
#             for j in range(len(vss)):
#                 count += 1
#                 hp = pos[0] + '_' + str(i)
#                 vp = pos[1] + '_' + str(j)
#                 prot2embed[hp] = to_onehot(hss[i], haaindex)
#                 prot2embed[vp] = to_onehot(vss[j], vaaindex)
#                 splited.append((hp, vp, pos[2]))
#         counts.append(count)
#     return counts
