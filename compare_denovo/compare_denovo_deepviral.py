import pickle
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
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from keras.utils import multi_gpu_model, Sequence, np_utils
import scipy.stats as ss
import sys
from utils import *
from models import *

MAXLEN = 1000
batch_size = 2000
params ={'max_kernel': 65, 'nb_filters': 16, 'pool_size': 200, 'dense_units': 8}
steps = 1500
epochs = 5

dataset = "denovo"
option = sys.argv[1]
virus_fasta = "data/AS2.fasta"
human_fasta = "data/AS3.fasta"
weights_file = f"weights.best.{dataset}.{option}.hdf5"
train_file = f'data/Additional_file_6.xlsx'
print(train_file)

unip2seq = dict()
vaaindex = get_index_fasta(virus_fasta, unip2seq)
haaindex = get_index_fasta(human_fasta, unip2seq)

embedding_file = sys.argv[2]
print(embedding_file)

taxon_file = "data/PPIs_Group.xlsx"
dfs_map = pd.read_excel(taxon_file, sheet_name = None)
vp2taxon = {}
for index, row in dfs_map["human"].iterrows(): 
    vp2taxon[row["VIRUS"]] = '<http://purl.obolibrary.org/obo/NCBITaxon_' + str(row["VIRUS_TAXID"]) + '>' 
print("len(vp2taxon)", len(vp2taxon))

with open("data/hpi.virus.family.txt", 'r') as f:
    for line in f:
        items = line.strip().split()
        vp2taxon[items[1]] = '<http://purl.obolibrary.org/obo/NCBITaxon_' + items[2] + '>'
print("len(vp2taxon)", len(vp2taxon))

data = pd.read_csv(embedding_file, header = None, sep = ' ', skiprows=1)
embds_data = data.values
embed_dict = dict(zip(embds_data[:,0],embds_data[:,1:-1]))
dim_input = 100
print('finished reading embeddings')

prot2embed = {}
tv_positives = []
tv_negatives = []
test_positives = []
test_negatives = []
dfs = pd.read_excel(train_file, sheet_name = None)

splits = ['positive training', 'negative training', 'positive test', 'negative test'] 
for split in splits:
    for index, row in dfs[split].iterrows():
        hp = row['HUMAN']
        vp = row['VIRUS']
        hs = unip2seq[hp]
        vs = unip2seq[vp]
        if (len(hs) > MAXLEN):
            hs = hs[:MAXLEN]
        if (len(vs) > MAXLEN):
            vs = vs[:MAXLEN]
        if option == "joint" and (hp not in embed_dict or vp not in vp2taxon or vp2taxon[vp] not in embed_dict):
            continue
        if option == 'viral' and (vp not in vp2taxon or vp2taxon[vp] not in embed_dict):
            continue
        if option == "human" and hp not in embed_dict:
            continue
        prot2embed[hp] = to_onehot(hs, haaindex)
        prot2embed[vp] = to_onehot(vs, vaaindex)
        if option in ['seq', 'human'] and vp not in vp2taxon:
            vp2taxon[vp] = 'dummy'
        if "training" in split:
            if 'positive' in split:
                tv_positives.append((hp,vp, vp2taxon[vp], 1))
            else:
                tv_negatives.append((hp,vp, vp2taxon[vp], 0))
        else:
            if 'positive' in split:
                test_positives.append((hp,vp, vp2taxon[vp], 1))
            else:
                test_negatives.append((hp,vp, vp2taxon[vp], 0))
                
print("len(tv_positives), len(tv_negatives), len(test_positives),len(test_negatives)", len(tv_positives), len(tv_negatives), len(test_positives),len(test_negatives))

recalls, specs, npvs, accs, precs, mccs, aucs, f1s = [], [], [], [],  [], [], [], []

for i in range(5):
    np.random.shuffle(tv_positives)
    np.random.shuffle(tv_negatives)

    split_factor = 0.9
    train_positives, val_positives = tv_positives[:int(len(tv_positives)*split_factor)],\
                                    tv_positives[int(len(tv_positives)*split_factor):]

    train_negatives, val_negatives = tv_negatives[:int(len(tv_negatives)*split_factor)],\
                                    tv_negatives[int(len(tv_negatives)*split_factor):]

    triple_train = np.concatenate((train_positives, train_negatives), axis=0)
    triple_val = np.concatenate((val_positives, val_negatives), axis=0)
    triple_test = np.concatenate((test_positives, test_negatives), axis=0)
    print("len(triple_train), len(triple_val), len(triple_test)", len(triple_train), len(triple_val), len(triple_test))
    
    if option =="seq":
        seq1, flat1 = get_seq_model(params, MAXLEN=MAXLEN)
        seq2, flat2 = get_seq_model(params, MAXLEN=MAXLEN)
        inputs = [seq1, seq2]
    elif option =="joint":
        seq1, pheno1, flat1 = get_joint_model(params)
        seq2, pheno2, flat2 = get_joint_model(params)
        inputs=[seq1, seq2, pheno1, pheno2]
    elif option =="viral":
        seq1, flat1 = get_seq_model(params)
        seq2, pheno2, flat2 = get_joint_model(params)
        flat2 = Dense(8)(flat2)
        flat2 = LeakyReLU(alpha=0.1)(flat2)
        flat2 = Dropout(0.5)(flat2)
        inputs=[seq1, seq2, pheno2]
    elif option =="human":
        seq1, pheno1, flat1 = get_joint_model(params)
        seq2, flat2 = get_seq_model(params)
        flat1 = Dense(8)(flat1)
        flat1 = LeakyReLU(alpha=0.1)(flat1)
        flat1 = Dropout(0.5)(flat1)
        inputs=[seq1, seq2, pheno1]

    concat = Dot(axes=-1, normalize=True)([flat1,flat2])
    output = Dense(1, activation='sigmoid', name='dense_out')(concat)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])

    train_gen, val_gen, test_gen = get_generators(triple_train, triple_val, triple_test, batch_size, prot2embed, option, embed_dict)

    checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose = 1, save_best_only=True, mode='max',\
                                save_weights_only=True)

    history = model.fit_generator(generator=train_gen,
                                  validation_data=val_gen,
                                  epochs=epochs,
                                  steps_per_epoch = steps,
                                  verbose=2,
                                  callbacks=[checkpoint])  
    model.load_weights(weights_file)

    y_pred = model.predict_generator(generator=test_gen, verbose=2)
    y_pred_label = np.zeros(len(y_pred))
    y_pred_label[np.where(y_pred.flatten() >= 0.5)] = 1
    y_pred_label[np.where(y_pred.flatten() < 0.5)] = 0

    y_true =triple_test[:,3].astype(np.int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()
    recall = recall_score(y_true, y_pred_label)
    spec = tn / (tn+fp)
    npv = tn/ (tn+fn)
    acc = accuracy_score(y_true, y_pred_label)
    prec = precision_score(y_true, y_pred_label)
    mcc = matthews_corrcoef(y_true, y_pred_label)
    auc = roc_auc_score(y_true, y_pred)
    f1 = 2 * prec * recall / (prec + recall)
    print("Sensitivity: %.4f, Specificity: %.4f, Accuracy: %.4f, PPV: %.4f, NPV: %.4f, MCC: %.4f, AUC: %.4f, F1: ROCAUC: %.4f" \
          % (recall*100, spec*100, acc*100, prec*100, npv*100, mcc, auc, f1*100))
    recalls.append(recall)
    specs.append(spec)
    npvs.append(npv)
    accs.append(acc)
    precs.append(prec)
    mccs.append(mcc)
    aucs.append(auc)
    f1s.append(f1)
              
print("Sensitivity: %.4f, Specificity: %.4f, Accuracy: %.4f, PPV: %.4f, NPV: %.4f, MCC: %.4f, AUC: %.4f, F1: ROCAUC: %.4f" \
          % (np.mean(recalls)*100, np.mean(specs)*100, np.mean(accs)*100, np.mean(precs)*100, np.mean(npvs)*100, np.mean(mccs), np.mean(aucs), np.mean(f1s)*100))
