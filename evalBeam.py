# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparameter import Hyperparameters as hp
from hyperparameter_local import HyperparametersLocal as hpl
from data_loadMS import *

from train_msconcat import Graph
from nltk.translate.bleu_score import corpus_bleu


def eval():
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")

    # Load data
    X1, X2, Sources1, Sources2, Targets = load_test_data()
    src2idx1, idx2src1 = load_src1_vocab()
    src2idx2, idx2src2 = load_src2_vocab()
    tgt2idx, idx2tgt = load_tgt_vocab()
    #    print(Sources)
    #     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
    logdir = hpl.logdir
    result_dir = hpl.result_dir + "/"

    nbest = 1

    # Start session
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(logdir))
            print("Restored!")
            print(logdir)
            ## Get model name
            mname = open(logdir + '/checkpoint', 'r').read().split('"')[1]
            if not os.path.exists(result_dir): os.mkdir(result_dir)
            with codecs.open(result_dir + mname, "w", "utf-8") as fout, codecs.open(result_dir + "reference", "w",
                                                                                    "utf-8") as fout1:
                list_of_refs, hypotheses = [], []
                for i in range(len(X1) // hp.batch_size):
                    ### Get mini-batches
                    x1 = X1[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources1 = Sources1[i * hp.batch_size: (i + 1) * hp.batch_size]

                    x2 = X2[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources2 = Sources2[i * hp.batch_size: (i + 1) * hp.batch_size]

                    targets = Targets[i * hp.batch_size: (i + 1) * hp.batch_size]
                    inputs1 = np.reshape(np.transpose(np.array([x1] * hp.beam_size), (1, 0, 2)),(hp.beam_size * hp.batch_size, hp.maxlen))
                    inputs2 = np.reshape(np.transpose(np.array([x2] * hp.beam_size), (1, 0, 2)),(hp.beam_size * hp.batch_size, hp.maxlen))

                    preds = np.zeros((hp.batch_size, hp.beam_size, hp.maxlen), np.int32)
                    prob_product = np.zeros((hp.batch_size, hp.beam_size))
                    stc_length = np.ones((hp.batch_size, hp.beam_size))

                    for j in range(hp.maxlen):
                        _probs, _preds = sess.run(
                            g.preds, {g.x1: inputs1, g.x2: inputs2, g.y: np.reshape(preds, (hp.beam_size * hp.batch_size, hp.maxlen))})
                        j_probs = np.reshape(_probs[:, j, :], (hp.batch_size, hp.beam_size, hp.beam_size))
                        j_preds = np.reshape(_preds[:, j, :], (hp.batch_size, hp.beam_size, hp.beam_size))
                        if j == 0:
                            preds[:, :, j] = j_preds[:, 0, :]
                            prob_product += np.log(j_probs[:, 0, :])
                        else:
                            add_or_not = np.asarray(np.logical_or.reduce([j_preds > hp.end_id]), dtype=np.int)
                            tmp_stc_length = np.expand_dims(stc_length, axis=-1) + add_or_not
                            tmp_stc_length = np.reshape(tmp_stc_length, (hp.batch_size, hp.beam_size * hp.beam_size))

                            this_probs = np.expand_dims(prob_product, axis=-1) + np.log(j_probs) * add_or_not
                            this_probs = np.reshape(this_probs, (hp.batch_size, hp.beam_size * hp.beam_size))
                            selected = np.argsort(this_probs / tmp_stc_length, axis=1)[:, -hp.beam_size:]

                            tmp_preds = np.concatenate([np.expand_dims(preds, axis=2)] * hp.beam_size, axis=2)
                            tmp_preds[:, :, :, j] = j_preds[:, :, :]
                            tmp_preds = np.reshape(tmp_preds, (hp.batch_size, hp.beam_size * hp.beam_size, hp.maxlen))

                            for batch_idx in range(hp.batch_size):
                                prob_product[batch_idx] = this_probs[batch_idx, selected[batch_idx]]
                                preds[batch_idx] = tmp_preds[batch_idx, selected[batch_idx]]
                                stc_length[batch_idx] = tmp_stc_length[batch_idx, selected[batch_idx]]

                    final_selected = np.argmax(prob_product / stc_length, axis=1)
                    final_preds = []
                    for batch_idx in range(hp.batch_size):
                        #print(preds[batch_idx, final_selected[batch_idx]])
                        got=""
                        for idx in preds[batch_idx, final_selected[batch_idx]]:
                            if idx < hp.maxlen:
                                got = " ".join(idx2tgt[idx])
                                got = got.split("</S>")[0].strip()
                                print(got)
                        target = targets[batch_idx]

                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
                        #final_preds.append(preds[batch_idx, final_selected[batch_idx]])
            score = corpus_bleu(list_of_refs, hypotheses)
            print("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
    eval()
    print("Done")