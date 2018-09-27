from __future__ import print_function
from hyperparameter_local import HyperparametersLocal as hp
import tensorflow as tf
import numpy as np
import codecs
import os
import regex
from collections import Counter


def make_vocab(infile, outfile):
    ''' create vocabulary.'''
    text = codecs.open(infile, 'r', 'utf-8').read()
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists(hp.prep): os.mkdir(hp.prep)
    with codecs.open(hp.prep + '/{}'.format(outfile), 'w', 'utf-8') as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == '__main__':
    make_vocab(hp.src1_train, hp.src1 + '.vocab.tsv')
    make_vocab(hp.src2_train, hp.src2 + '.vocab.tsv')
    make_vocab(hp.tgt_train, hp.tgt + '.vocab.tsv')
