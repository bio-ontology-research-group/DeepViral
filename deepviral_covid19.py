import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

from keras.models import Model, load_model
from keras.layers import (
    Input, Dense, Embedding, Conv1D, Flatten, Concatenate,
    MaxPooling1D, Dropout, Dot, LeakyReLU
)
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score
from keras.utils import multi_gpu_model, Sequence, np_utils
import scipy.stats as ss

from utils import *
from models import *

MAXLEN = 1000
epochs = 5
num_gpus = 1
batch_size = 1500*num_gpus
steps = 1000

thres = '0'
option = sys.argv[1]
tid = sys.argv[3]
embedding_file = sys.argv[2]
print("option: ", option, "threshold: ", thres)
corona_interactions = 'data/media-6.xlsx'
corona_sequences = 'data/2020-04-krogan-sarscov2-sequences-uniprot-mapping.xlsx'

model_file = f'model_covid19_{option}_{tid}.h5'
preds_file = f'preds_covid19_{option}_{tid}.txt'
open_preds = open(preds_file, "w")
open_preds.close()

swissprot_file = 'data/swissprot-proteome.tab'
hpi_file = 'data/train_1000.txt'

params = get_params()

haaindex, vaaindex = get_aaindex(swissprot_file, hpi_file)
print("The amino acids for human are ", list(haaindex))
print("The amino acids for viruses are ", list(vaaindex))
embed_dict = read_embedding(embedding_file)
hp_set, prot2embed = read_swissprot(swissprot_file, embed_dict, haaindex, first = False)

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
            if family == '<http://purl.obolibrary.org/obo/NCBITaxon_11118>':
                continue
            prot2embed[vp] = to_onehot(items[5], vaaindex)
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
    
vp_set = set(vp2patho.keys())
families = set(family2vp.keys())
print('Number of positives: ', len(positives))
print('Number of pathogens: ', len(pathogens))
print('Number of families: ', len(families))
print('Number of viral proteins: ', len(vp_set))

dfs_seq = pd.read_excel(corona_sequences)
for index, row in dfs_seq.iterrows():
    vp = row['Krogan name']
    seq = row['Sequence']
    if '*' == seq[len(seq)-1]:
        seq = seq[:len(seq)-1]
    prot2embed[vp] = to_onehot(seq, vaaindex)
    
config = tf.ConfigProto()
sess = tf.Session(config=config)
K.set_session(sess)

counter = 0
family_aucs = []
for test_family in ["<http://purl.obolibrary.org/obo/NCBITaxon_11118>"]:
    counter+=1
    K.clear_session()
    print('Test family %d: %s' % (counter, test_family))
    tv_families = list(families - set([test_family]))
    val_families = set(np.random.choice(tv_families, size = int(len(tv_families)/5), replace=False))
    train_families = set(tv_families) - val_families
    print('Train families: ', len(train_families), 'validation families', len(val_families))

    train_vps = set()
    for family in train_families:
        train_vps = train_vps | family2vp[family]
    val_vps = vp_set - family2vp[test_family] - train_vps
    print("Number of viral proteins in train, val and test: ", len(train_vps), len(val_vps), len(family2vp[test_family]))
    
    triple_train = get_triple(positives, train_families, hp_set, train_vps, vp2patho, 'train')
    triple_val, numPos_val = get_triple(positives, val_families, hp_set, val_vps,  vp2patho, 'val')
    triple_test, numPos_test = get_triple(positives, [test_family], hp_set, family2vp[test_family], vp2patho, 'test')
    print("Number of triples in train, val, test", len(triple_train), len(triple_val), len(triple_test))
    
    if option =="seq":
        seq1, flat1 = get_seq_model(params)
        seq2, flat2 = get_seq_model(params)
    if option =="joint":
        seq1, pheno1, flat1 = get_joint_model(params)
        seq2, pheno2, flat2 = get_joint_model(params)
    if option == "pheno":
        seq1, flat1 = get_seq_model(params)
        seq2, pheno2, flat2 = get_joint_model(params)
        flat2 = Dense(8)(flat2)
        flat2 = LeakyReLU(alpha=0.1)(flat2)
        flat2 = Dropout(0.5)(flat2)
    if option == "go":
        seq1, pheno1, flat1 = get_joint_model(params)
        seq2, flat2 = get_seq_model(params)
        flat1 = Dense(8)(flat1)
        flat1 = LeakyReLU(alpha=0.1)(flat1)
        flat1 = Dropout(0.5)(flat1)
    concat = Dot(axes=-1, normalize=True)([flat1,flat2])
    output = Dense(1, activation='sigmoid')(concat)

    if option =="seq":
        model = Model(inputs=[seq1, seq2], outputs=output)
    if option =="joint":
        model = Model(inputs=[seq1, seq2, pheno1, pheno2], outputs=output)
    if option == "pheno":
        model = Model(inputs=[seq1, seq2, pheno2], outputs=output)
    if option == "go":
        model = Model(inputs=[seq1, seq2, pheno1], outputs=output)
    train_gen, val_gen, test_gen = get_generators(triple_train, triple_val, triple_test, batch_size, prot2embed, option, embed_dict)
        
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])

    val_maxauc = 0
    for i in range(epochs):
        print('taxon ', counter, ' epoch ', i)
        history = model.fit_generator(generator=train_gen,
                            epochs=1,
                            steps_per_epoch = steps,
                            verbose=2,
                            max_queue_size = 50,
                            use_multiprocessing=False,
                            workers = 1)

        y_score = model.predict_generator(generator=val_gen, verbose=2,
                                           steps=int(np.ceil(len(triple_val)/batch_size)), 
                                            max_queue_size = 50, workers = 1)
            
        y_true = np.concatenate((np.ones(numPos_val), np.zeros(len(triple_val) - numPos_val)))
        
        val_auc = roc_auc_score(y_true, y_score)
        print('The ROCAUC for the val families in this epoch is ', val_auc)
        if val_auc > val_maxauc:
            print('Saving current model...')
            model.save(model_file)
            val_maxauc = val_auc

    del model
    K.clear_session()
    model = load_model(model_file)
    y_score = model.predict_generator(generator=test_gen,
                                    verbose=2,steps=np.ceil(len(triple_test)/batch_size), 
                                    max_queue_size = 30, use_multiprocessing=False, workers = 1)

    y_true = np.concatenate((np.ones(numPos_test), np.zeros(len(triple_test) - numPos_test)))
    auprc = average_precision_score(y_true, y_score)
    test_auc = roc_auc_score(y_true, y_score)
    family_aucs.append((test_auc))
    print("ROCAUC: %.4f, AUPRC: %.4f" % (test_auc, auprc))
    
    with open(preds_file, 'a+') as f:
        for i in range(triple_test.shape[0]):
            f.write("%s\t%s\t%s\t%s\t%f\t%s\n" % (triple_test[i,1], triple_test[i,0], triple_test[i,2], test_family, y_score[i], i<numPos_test))
    
print("Mean ROCAUC of test families: %.4f" % (np.mean(family_aucs)))


        
#         y_true = y_true[:2*numPos_test]
#         y_score_pos = y_score[:numPos_test]
#         y_score_neg = y_score[numPos_test:]
#         np.random.shuffle(y_score_neg)
#         y_score = np.concatenate((y_score_pos, y_score_neg[:numPos_test]))
        
#         y_pred_label = np.zeros(len(y_score))
#         y_pred_label[np.where(y_score.flatten() >= 0.5)] = 1
#         y_pred_label[np.where(y_score.flatten() < 0.5)] = 0

#         acc = accuracy_score(y_true, y_pred_label)
#         prec = precision_score(y_true, y_pred_label)
#         recall = recall_score(y_true, y_pred_label)

#         print("Accuracy: %.4f, Precision: %.4f, Recall: %.4f, ROCAUC: %.4f, AUPRC: %.4f" % (acc*100, prec*100, recall*100, test_auc, auprc))
        
