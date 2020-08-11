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

def get_generators(triple_train, triple_val, triple_test, batch_size, prot2embed, option, embed_dict=None, MAXLEN=2000):
    if option == "seq":
        generator = seq_Generator(triple_train[:,0:2], triple_train[:,3], batch_size, prot2embed, MAXLEN)
        val_generator = seq_Generator(triple_val[:,0:2], triple_val[:,3], batch_size, prot2embed, MAXLEN)
        test_generator = seq_Generator(triple_test[:,0:2], triple_test[:,3], batch_size, prot2embed, MAXLEN)
    elif option == "joint":
        generator = joint_Generator(triple_train[:,0:3], triple_train[:,3], batch_size, prot2embed, embed_dict)
        val_generator = joint_Generator(triple_val[:,0:3], triple_val[:,3], batch_size, prot2embed, embed_dict)
        test_generator = joint_Generator(triple_test[:,0:3], triple_test[:,3], batch_size, prot2embed, embed_dict)
    elif option == "pheno":
        generator = pheno_Generator(triple_train[:,0:3], triple_train[:,3], batch_size, prot2embed, embed_dict)
        val_generator = pheno_Generator(triple_val[:,0:3], triple_val[:,3], batch_size, prot2embed, embed_dict)
        test_generator = pheno_Generator(triple_test[:,0:3], triple_test[:,3], batch_size, prot2embed, embed_dict)
    return generator, val_generator, test_generator

def get_seq_model(params,MAXLEN = 1000):
    seq = Input(shape=(MAXLEN, 22), dtype=np.float32)
    kernels = range(8, params['max_kernel'], 8)
    nets = []
    for i in range(len(kernels)):
        conv = Conv1D(
            filters=params['nb_filters'],
            kernel_size=kernels[i],
            padding='valid',
            kernel_initializer='glorot_normal')(seq)
        conv_dropout = Dropout(0.5)(conv)
        pool = MaxPooling1D(pool_size=params['pool_size'])(conv_dropout)
        flat = Flatten()(pool)
        nets.append(flat)

    net = Concatenate(axis=1)(nets)
    dense_seq = Dense(params['dense_units'])(net)
    activ_seq = LeakyReLU(alpha=0.1)(dense_seq)
    dropout_seq = Dropout(0.5)(activ_seq)

    return seq, dropout_seq

class seq_Generator(Sequence):
    def __init__(self, x_set, y_set, batch_size, prot2embed, MAXLEN = 1000):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.nbatch = int(np.ceil(len(self.x) / float(self.batch_size)))
        self.length = len(self.x)
        self.prot2embed = prot2embed
        self.MAXLEN = MAXLEN

    def __len__(self):
        return self.nbatch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        batch_len = min(self.batch_size, (self.length)-start)
        X_batch_list1 = np.empty((batch_len, self.MAXLEN, 13), dtype=np.float32)
        X_batch_list2 = np.empty((batch_len, self.MAXLEN, 13), dtype=np.float32)
        y_batch_list = np.empty(batch_len, dtype=np.float32)

        for ids in range(start, min((idx + 1) * self.batch_size, self.length)):
            array1 = self.prot2embed[self.x[ids][0]]
            array2 = self.prot2embed[self.x[ids][1]]
            X_batch_list1[ids-start,:,:] = array1
            X_batch_list2[ids-start,:,:] = array2
            y_batch_list[ids-start] = self.y[ids]
        return [X_batch_list1,X_batch_list2], y_batch_list

def get_joint_model(params,MAXLEN = 1000):
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

class joint_Generator(Sequence):
    def __init__(self, x_set, y_set, batch_size, prot2embed, embed_dict):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.nbatch = int(np.ceil(len(self.x) / float(self.batch_size)))
        self.length = len(self.x)
        self.prot2embed = prot2embed
        self.embed_dict = embed_dict
        self.dim_input = 100
        
    def __len__(self):
        return self.nbatch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        batch_len = min(self.batch_size, (self.length)-start)
        x_seq1 = np.empty((batch_len, 1000,22), dtype=np.float32)
        x_seq2 = np.empty((batch_len, 1000,22), dtype=np.float32)
        x_pheno1 = np.empty((batch_len, self.dim_input), dtype=np.float32)
        x_pheno2 = np.empty((batch_len, self.dim_input), dtype=np.float32)
        y_batch = np.empty(batch_len, dtype=np.float32)

        for ids in range(start, min((idx + 1) * self.batch_size, self.length)):
            x_seq1[ids-start,:,:] = self.prot2embed[self.x[ids][0]]
            x_seq2[ids-start,:,:] = self.prot2embed[self.x[ids][1]]
            x_pheno1[ids-start,:] = self.embed_dict[self.x[ids][0]]
            x_pheno2[ids-start,:] = self.embed_dict[self.x[ids][2]]
            y_batch[ids-start] = self.y[ids]
        return [x_seq1, x_seq2, x_pheno1, x_pheno2], y_batch
    
class pheno_Generator(Sequence):
    def __init__(self, x_set, y_set, batch_size, prot2embed, embed_dict):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.nbatch = int(np.ceil(len(self.x) / self.batch_size))
        self.length = len(self.x)
        self.prot2embed = prot2embed
        self.embed_dict = embed_dict
        self.dim_input = 100
        
    def __len__(self):
        return self.nbatch

    def __getitem__(self, idx):
        start = idx * self.batch_size
        batch_len = min(self.batch_size, (self.length)-start)
        x_seq1 = np.empty((batch_len, 1000,22), dtype=np.float32)
        x_seq2 = np.empty((batch_len, 1000,22), dtype=np.float32)
        x_pheno2 = np.empty((batch_len, self.dim_input), dtype=np.float32)
        y_batch = np.empty(batch_len, dtype=np.float32)

        for ids in range(start, min((idx + 1) * self.batch_size, self.length)):
            x_seq1[ids-start,:,:] = self.prot2embed[self.x[ids][0]]
            x_seq2[ids-start,:,:] = self.prot2embed[self.x[ids][1]]
            x_pheno2[ids-start,:] = self.embed_dict[self.x[ids][2]]
            y_batch[ids-start] = self.y[ids]
        return [x_seq1, x_seq2, x_pheno2], y_batch