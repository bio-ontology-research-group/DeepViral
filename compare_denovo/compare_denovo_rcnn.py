from seq2tensor import s2t

import keras
from keras.models import Sequential, Model
from keras.utils import multi_gpu_model, Sequence, np_utils
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, merge, add
from keras.layers.core import Flatten, Reshape
from keras.layers.merge import Concatenate, concatenate, subtract, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers import Input, CuDNNGRU
from keras.optimizers import Adam,  RMSprop
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
import sys
from tqdm import tqdm
from numpy import linalg as LA
import scipy
import numpy as np
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from utils import *
from models import *

seq_size = 2000
MAXLEN = seq_size
seq2t = s2t('vec5_CTC.txt')
hidden_dim = 50
dim = seq2t.dim

epochs = 5
num_gpus = 1
batch_size = 2000*num_gpus
steps = 500
verbose = 1

weights_file = f'model_rcnn_denovo.h5'
swissprot_file = 'data/swissprot-proteome.tab'
hpi_file = 'data/train_1000.txt'

dataset = "denovo"
virus_fasta = "data/AS2.fasta"
human_fasta = "data/AS3.fasta"
train_file = f'data/Additional_file_6.xlsx'
print(train_file)

unip2seq = dict()
vaaindex = get_index_fasta(virus_fasta, unip2seq)
haaindex = get_index_fasta(human_fasta, unip2seq)

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
        prot2embed[hp] = np.array(seq2t.embed_normalized(hs, seq_size))
        prot2embed[vp] = np.array(seq2t.embed_normalized(vs, seq_size))
        if "training" in split:
            if 'positive' in split:
                tv_positives.append((hp,vp, 1))
            else:
                tv_negatives.append((hp,vp, 0))
        else:
            if 'positive' in split:
                test_positives.append((hp,vp, 1))
            else:
                test_negatives.append((hp,vp, 0))
                
print("len(tv_positives), len(tv_negatives), len(test_positives),len(test_negatives)", len(tv_positives), len(tv_negatives), len(test_positives),len(test_negatives))

def build_model():
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    l1=Conv1D(hidden_dim, 3)
    r1=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l2=Conv1D(hidden_dim, 3)
    r2=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l3=Conv1D(hidden_dim, 3)
    r3=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l4=Conv1D(hidden_dim, 3)
    r4=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l5=Conv1D(hidden_dim, 3)
    r5=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l6=Conv1D(hidden_dim, 3)
    s1=MaxPooling1D(3)(l1(seq_input1))
    s1=concatenate([r1(s1), s1])
    s1=MaxPooling1D(3)(l2(s1))
    s1=concatenate([r2(s1), s1])
    s1=MaxPooling1D(3)(l3(s1))
    s1=concatenate([r3(s1), s1])
    s1=MaxPooling1D(3)(l4(s1))
    s1=concatenate([r4(s1), s1])
    s1=MaxPooling1D(3)(l5(s1))
    s1=concatenate([r5(s1), s1])
    s1=l6(s1)
    s1=GlobalAveragePooling1D()(s1)
    s2=MaxPooling1D(3)(l1(seq_input2))
    s2=concatenate([r1(s2), s2])
    s2=MaxPooling1D(3)(l2(s2))
    s2=concatenate([r2(s2), s2])
    s2=MaxPooling1D(3)(l3(s2))
    s2=concatenate([r3(s2), s2])
    s2=MaxPooling1D(3)(l4(s2))
    s2=concatenate([r4(s2), s2])
    s2=MaxPooling1D(3)(l5(s2))
    s2=concatenate([r5(s2), s2])
    s2=l6(s2)
    s2=GlobalAveragePooling1D()(s2)
    merge_text = multiply([s1, s2])
    x = Dense(100, activation='linear')(merge_text)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(int((hidden_dim+7)/2), activation='linear')(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(1, activation='sigmoid')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model

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
    
    model = None
    model = build_model()
    adam = Adam()
    rms = RMSprop(lr=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    train_gen = seq_Generator_rcnn(triple_train[:,0:2], triple_train[:,2], batch_size, prot2embed, MAXLEN=MAXLEN)
    val_gen = seq_Generator_rcnn(triple_val[:,0:2], triple_val[:,2], batch_size, prot2embed, MAXLEN=MAXLEN)
    test_gen = seq_Generator_rcnn(triple_test[:,0:2], triple_test[:,2], batch_size, prot2embed, MAXLEN=MAXLEN)
    
    checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose = 1, save_best_only=True, mode='max',\
                                save_weights_only=True)

    history = model.fit_generator(generator=train_gen,
                                  validation_data=val_gen,
                                  epochs=epochs,
                                  steps_per_epoch = steps,
                                  verbose=verbose,
                                  callbacks=[checkpoint])  
    model.load_weights(weights_file)

    y_pred = model.predict_generator(generator=test_gen, verbose=verbose)
    y_pred_label = np.zeros(len(y_pred))
    y_pred_label[np.where(y_pred.flatten() >= 0.5)] = 1
    y_pred_label[np.where(y_pred.flatten() < 0.5)] = 0

    y_true =triple_test[:,2].astype(np.int)

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
