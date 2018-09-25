from __future__ import print_function
from hyperparameter_local import HyperparametersLocal as hpl
from hyperparameter import Hyperparameters as hp
import tensorflow as tf
import numpy as np
import codecs
import os
import regex

def load_src1_vocab():
    vocab = [line.split()[0] for line in codecs.open(hpl.prep+'/'+ hpl.src1+'.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word
    
def load_src2_vocab():
    vocab = [line.split()[0] for line in codecs.open(hpl.prep+'/'+ hpl.src2+'.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word    

def load_tgt_vocab():
    vocab = [line.split()[0] for line in codecs.open(hpl.prep+'/'+ hpl.tgt+'.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents1, source_sents2, target_sents): 
    src2idx1, idx2src1 = load_src1_vocab()
    src2idx2, idx2src2 = load_src2_vocab()
    tgt2idx, idx2tgt = load_tgt_vocab()
    
    # Index
    x1_list, x2_list, y_list, Sources1, Sources2, Targets = [], [], [], [], [], []
    for source_sent1, source_sent2, target_sent in zip(source_sents1, source_sents2, target_sents):
        x1 = [src2idx1.get(word, 1) for word in (source_sent1 + u" </S>").split()] # 1: OOV, </S>: End of Text
        x2 = [src2idx2.get(word, 1) for word in (source_sent2 + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [tgt2idx.get(word, 1) for word in (target_sent + u" </S>").split()] 
        if max(len(x1), len(x2), len(y)) <=hp.maxlen:
            x1_list.append(np.array(x1))
            x2_list.append(np.array(x2))
            y_list.append(np.array(y))
            Sources1.append(source_sent1)
            Sources2.append(source_sent2)
            Targets.append(target_sent)
    
    # Pad      
    X1 = np.zeros([len(x1_list), hp.maxlen], np.int32)
    X2 = np.zeros([len(x2_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x1, x2, y) in enumerate(zip(x1_list, x2_list, y_list)):
        X1[i] = np.lib.pad(x1, [0, hp.maxlen-len(x1)], 'constant', constant_values=(0, 0))
        X2[i] = np.lib.pad(x2, [0, hp.maxlen-len(x2)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    return X1, X2, Y, Sources1, Sources2, Targets

def load_train_data():
    src_sents1 = [line for line in codecs.open(hpl.src1_train, 'r', 'utf-8').read().split("\n")]
    src_sents2 = [line for line in codecs.open(hpl.src2_train, 'r', 'utf-8').read().split("\n")]
    tgt_sents = [line for line in codecs.open(hpl.tgt_train, 'r', 'utf-8').read().split("\n")]
    
    X1, X2, Y, Sources1, Sources2, Targets = create_data(src_sents1, src_sents2, tgt_sents)
    return X1, X2, Y

def load_test_data():
    src_sents1 = [line for line in codecs.open(hpl.src1_test, 'r', 'utf-8').read().split("\n")]
    src_sents2 = [line for line in codecs.open(hpl.src2_test, 'r', 'utf-8').read().split("\n")]
    tgt_sents = [line for line in codecs.open(hpl.tgt_test, 'r', 'utf-8').read().split("\n")]
        
    X1, X2, Y, Sources1, Sources2, Targets = create_data(src_sents1, src_sents2, tgt_sents)
    return X1, X2, Sources1, Sources2, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X1, X2, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X1) // hp.batch_size
    print('Number of Batch: ', num_batch)
    
    # Convert to tensor
    X1 = tf.convert_to_tensor(X1, tf.int32)
    X2 = tf.convert_to_tensor(X2, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X1, X2, Y])
            
    # create batch queues
    x1, x2, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    print(num_batch)
    return x1, x2, y, num_batch # (N, T), (N, T), ()
