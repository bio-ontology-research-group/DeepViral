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
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from keras.utils import multi_gpu_model, Sequence, np_utils
import scipy.stats as ss
import sys

MAXLEN = 1000
batch_size = 2000
params ={'max_kernel': 65, 'nb_filters': 16, 'pool_size': 200, 'dense_units': 8}
steps = 1000

virus = sys.argv[1]
dataset = sys.argv[2]
weights_file = f"weights.best.{virus}.{dataset}.hdf5"
train_file = f'data/{virus}/{dataset}/Train.xlsx'
test_file = f'data/{virus}/{dataset}/Test.xlsx'
print(train_file, test_file)

def repeat_to_length(s, length):
    return (s * (length//len(s) + 1))[:length]

def to_onehot(seq, option, start=0):
    onehot = np.zeros((MAXLEN, 22), dtype=np.int32)
    seq = repeat_to_length(seq, MAXLEN)
    l = min(MAXLEN, len(seq))
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
        conv_dropout = Dropout(0.5)(conv)
        pool = MaxPooling1D(pool_size=params['pool_size'])(conv_dropout)
        flat = Flatten()(pool)
        nets.append(flat)

    net = Concatenate(axis=1)(nets)
    dense_seq = Dense(params['dense_units'])(net)
    activ_seq = LeakyReLU(alpha=0.1)(dense_seq)
    dropout_seq = Dropout(0.5)(activ_seq)

    return seq, dropout_seq

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
        X_batch_list1 = np.empty((batch_len, 1000,22), dtype=np.float32)
        X_batch_list2 = np.empty((batch_len, 1000,22), dtype=np.float32)
        y_batch_list = np.empty(batch_len, dtype=np.float32)

        for ids in range(start, min((idx + 1) * self.batch_size, self.length)):
            array1 = prot2embed[self.x[ids][0]]
            array2 = prot2embed[self.x[ids][1]]
            X_batch_list1[ids-start,:,:] = array1
            X_batch_list2[ids-start,:,:] = array2
            y_batch_list[ids-start] = self.y[ids]
        return [X_batch_list1,X_batch_list2], y_batch_list



prot2embed = {}
tv_positives = []
tv_negatives = []
test_positives = []
test_negatives = []

dfs_train = pd.read_excel(train_file, sheet_name = None)
dfs_test = pd.read_excel(test_file, sheet_name = None)

splits = ['Train_POS', 'Train_NEG'] 
for split in splits:
    for index, row in dfs_train[split].iterrows():
        hp = row['HOST']
        vp = row['VIRUS']
        hs = row['HOST_SEQ']
        vs = row['VIRUS_SEQ']
        if (len(hs) > MAXLEN) or (len(vs) > MAXLEN):
            continue
        prot2embed[hp] = to_onehot(hs, 'human')
        prot2embed[vp] = to_onehot(vs, 'virus')
        if 'POS' in split:
            tv_positives.append((hp,vp,1))
        else:
            tv_negatives.append((hp,vp,0))

splits = ['Test_POS', 'Test_NEG']
for split in splits:    
    for index, row in dfs_test[split].iterrows():
        hp = row['HOST']
        vp = row['VIRUS']
        hs = row['HOST_SEQ']
        vs = row['VIRUS_SEQ']
        if (len(hs) > MAXLEN) or (len(vs) > MAXLEN):
            continue
        prot2embed[hp] = to_onehot(hs, 'human')
        prot2embed[vp] = to_onehot(vs, 'virus')
        if 'POS' in split:
            test_positives.append((hp,vp,1))
        else:
            test_negatives.append((hp,vp,0))

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
print(len(triple_train), len(triple_val), len(triple_test))

seq1, flat1 = get_model()
seq2, flat2 = get_model()

concat = Dot(axes=-1, normalize=True)([flat1,flat2])
output = Dense(1, activation='sigmoid', name='dense_out')(concat)

model = Model(inputs=[seq1, seq2], outputs=output)
#model.summary()
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy'])

generator = Generator(triple_train[:,0:2], triple_train[:,2], batch_size)
val_generator = Generator(triple_val[:,0:2], triple_val[:,2], batch_size)

checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose = 1, save_best_only=True, mode='max',\
                            save_weights_only=True)
history = model.fit_generator(generator=generator,
                              validation_data=val_generator,
                              epochs=5,
                              steps_per_epoch = steps,
                              verbose=2,
                              callbacks=[checkpoint])  

model.load_weights(weights_file)

test_generator = Generator(triple_test[:,0:2], triple_test[:,2], batch_size)
y_pred = model.predict_generator(generator=test_generator)

y_pred_label = np.zeros(len(y_pred))
y_pred_label[np.where(y_pred.flatten() >= 0.5)] = 1
y_pred_label[np.where(y_pred.flatten() < 0.5)] = 0

y_true =triple_test[:,2].astype(np.int)
acc = accuracy_score(y_true, y_pred_label)
prec = precision_score(y_true, y_pred_label)
recall = recall_score(y_true, y_pred_label)
auc = roc_auc_score(y_true, y_pred)
print("Accuracy: %.4f, Precision: %.4f, Recall: %.4f, ROCAUC: %.4f" % (acc, prec, recall, auc))
print("& \\textbf{%.2f} & \\textbf{%.2f} & \\textbf{%.2f} & \\textbf{%.3f}" % (acc*100, prec*100, recall*100, auc))
