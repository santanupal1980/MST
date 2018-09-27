# -*- coding: utf-8 -*-
# /usr/bin/python2
from __future__ import print_function
import tensorflow as tf

from hyperparameter import Hyperparameters as hp
from hyperparameter_local import HyperparametersLocal as hpl
from data_loadMS import *
from model import *
import os, codecs
from tqdm import tqdm
import numpy as np
#from keras.models import *


class Graph():

    def encoder(self, x, vocab_size, scope=None, is_training=False):

        enc = embedding(x,
                        vocab_size=vocab_size,
                        num_units=hp.hidden_units,
                        scale=True,
                        scope=scope + "_embed")

        ## Positional Encoding
        if hp.sinusoid:
            enc += positional_encoding(x,
                                       num_units=hp.hidden_units,
                                       zero_pad=False,
                                       scale=False,
                                       scope=scope + "_pe")
        else:
            enc += embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1]),
                vocab_size=hp.maxlen,
                num_units=hp.hidden_units,
                zero_pad=False,
                scale=False,
                scope=scope + "_pe")

        ## Dropout
        enc = tf.layers.dropout(enc,
                                rate=hp.dropout_rate,
                                training=tf.convert_to_tensor(is_training))

        ## Blocks
        for i in range(hp.num_blocks):
            with tf.variable_scope("enc1_num_blocks_{}".format(i)):
                ### Multihead Attention
                enc_att, _ = multihead_attention(queries=enc,
                                             keys=enc,
                                             num_units=hp.hidden_units,
                                             num_heads=hp.num_heads,
                                             dropout_rate=hp.dropout_rate,
                                             is_training=is_training,
                                             causality=False)

                ### Feed Forward
                enc += feedforward(enc_att, num_units=[4 * hp.hidden_units, hp.hidden_units])
        return enc

    def decoder(self, decoder_inputs, vocab_size, enc, scope=None, is_training=False):
        dec = embedding(decoder_inputs,
                        vocab_size=vocab_size,
                        num_units=hp.hidden_units,
                        scale=True,
                        scope=scope + "_embed")

        ## Positional Encoding
        if hp.sinusoid:
            dec += positional_encoding(decoder_inputs,
                                       num_units=hp.hidden_units,
                                       zero_pad=False,
                                       scale=False,
                                       scope=scope + "_pe")
        else:
            dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(decoder_inputs)[1]), 0),
                                     [tf.shape(decoder_inputs)[0], 1]),
                             vocab_size=hp.maxlen,
                             num_units=hp.hidden_units,
                             zero_pad=False,
                             scale=False,
                             scope="dec_pe")

        ## Dropout
        dec = tf.layers.dropout(dec,
                                rate=hp.dropout_rate,
                                training=tf.convert_to_tensor(is_training))

        ## Blocks
        for i in range(hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ## Multihead Attention ( self-attention)
                dec_satt, _ = multihead_attention(queries=dec,
                                             keys=dec,
                                             num_units=hp.hidden_units,
                                             num_heads=hp.num_heads,
                                             dropout_rate=hp.dropout_rate,
                                             is_training=is_training,
                                             causality=True,
                                             scope="self_attention")

                ## Multihead Attention ( vanilla attention)
                dec_vatt, alignments = multihead_attention(queries=dec_satt,
                                                      keys=enc,
                                                      num_units=hp.hidden_units,
                                                      num_heads=hp.num_heads,
                                                      dropout_rate=hp.dropout_rate,
                                                      is_training=is_training,
                                                      causality=False,
                                                      scope="vanilla_attention")

                ## Feed Forward
                dec += feedforward(dec_vatt, num_units=[4 * hp.hidden_units, hp.hidden_units])

        return dec, alignments

    def Joint_encoder(self, enc1, enc2, is_training=False):
        enc = tf.concat([enc1, enc2], 1)
        ## Dropout
        enc = tf.layers.dropout(enc,
                                rate=hp.dropout_rate,
                                training=tf.convert_to_tensor(is_training))
        ## Blocks
        for i in range(hp.num_blocks):
            with tf.variable_scope("enc_num_blocks_{}".format(i)):
                ### Multihead Attention
                enc_att, _ = multihead_attention(queries=enc,
                                             keys=enc,
                                             num_units=hp.hidden_units,
                                             num_heads=hp.num_heads,
                                             dropout_rate=hp.dropout_rate,
                                             is_training=is_training,
                                             causality=False)

                ### Feed Forward
                enc += feedforward(enc_att, num_units=[4 * hp.hidden_units, hp.hidden_units])
        return enc



    def __init__(self, is_training=True):
        # Load vocabulary

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.decode_model = None
            if is_training:
                self.x1, self.x2, self.y, self.num_batch = get_batch_data()  # (N, T) Creating batch input
            else:  # inference
                self.x1 = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.x2 = tf.placeholder(tf.int32, shape=(None, hp.maxlen))


                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)  # 2:<S>

            src2idx1, idx2src1 = load_src1_vocab()
            src2idx2, idx2src2 = load_src2_vocab()
            tgt2idx, idx2tgt = load_tgt_vocab()
            print('Vocab loaded')

            # Encoder1
            with tf.variable_scope("encoder1"):
                self.enc1 = self.encoder(self.x1, len(src2idx1), scope='enc1', is_training=is_training)
            # Encoder2
            with tf.variable_scope("encoder2"):
                self.enc2 = self.encoder(self.x2, len(src2idx2), scope='enc2', is_training=is_training)

            # Joint Encoder
            with tf.variable_scope("encoder"):
                self.enc = self.Joint_encoder(self.enc1, self.enc2, is_training=is_training)

            # Decoder

            with tf.variable_scope("decoder"):
                self.dec, self.alignments = self.decoder(self.decoder_inputs, len(tgt2idx), self.enc, scope="dec",
                                                         is_training=is_training)


            # Final linear projection
            self.logits = tf.layers.dense(self.dec, len(tgt2idx))
            self.preds_trn = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds_trn, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)

            # Loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(tgt2idx)))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
            self.mean_loss = tf.reduce_mean(self.mean_loss)
            self.perplexity = np.power(2, self.mean_loss)

            # Training Scheme
            self.global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                               trainable=False, initializer=tf.zeros_initializer)
            learning_rate_warmup_steps = 4000
            warmup_steps = tf.to_float(learning_rate_warmup_steps)
            global_step = tf.to_float(self.global_step)
            learning_rate = hp.hidden_units ** -0.5 * tf.minimum(
                (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)

            #learning_rate = hp.lr #tf.train.exponential_decay(hp.lr, self.global_step, 100000, 0.96, staircase=True)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,
                                                    epsilon=1e-8)
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_locking = False, name = 'Momentum', use_nesterov = True)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            tf.summary.scalar('perplexity', self.perplexity)
            self.merged = tf.summary.merge_all()


            if not is_training:
                if hp.is_greedy:
                    self.preds= tf.to_int32(tf.arg_max(self.logits, dimension=-1))
                else:
                    print('[WARNING] beam search enabled')
                    assert self.logits.get_shape().as_list()[-1] >= hp.beam_size
                    self.probs = tf.nn.softmax(self.logits)
                    self.preds = tf.nn.top_k(self.probs, hp.beam_size)






if __name__ == '__main__':
    # Load vocabulary
    src2idx1, idx2src1 = load_src1_vocab()
    src2idx2, idx2src2 = load_src2_vocab()
    tgt2idx, idx2tgt = load_tgt_vocab()

    # Construct graph
    g = Graph("train")
    print("Graph loaded")

    X1, X2, Y = load_train_data()

    # calc total batch count
    num_batch = len(X1) // hp.batch_size
    print(X1.shape)
    g.num_batch = num_batch
    # Start session
    sv = tf.train.Supervisor(graph=g.graph,
                             logdir=hpl.logdir,
                             summary_op=None,
                             save_model_secs=0)
    with sv.managed_session() as sess:
        i = 0
        for epoch in range(1, hp.num_epochs + 1):
            print(epoch)
            if sv.should_stop(): break

            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                # sess.run(g.train_op)
                x1 = X1[step * hp.batch_size:(step + 1) * hp.batch_size]
                x2 = X2[step * hp.batch_size:(step + 1) * hp.batch_size]

                y = Y[step * hp.batch_size:(step + 1) * hp.batch_size]
                x1 = np.array(x1, dtype=np.int32)
                x2 = np.array(x2, dtype=np.int32)
                y = np.array(y, dtype=np.int32)
                sess.run([g.train_op, g.merged], {g.x1: x1, g.x2: x2, g.y: y})
                i += 1
                if step % 100 == 0:
                    sv.summary_computed(sess, sess.run(g.merged, {g.x1: x1, g.x2: x2, g.y: y}))
                    _preds, _alignments, _gs = sess.run([g.preds_trn, g.alignments, g.global_step],
                                                        {g.x1: x1, g.x2: x2, g.y: y})
                    # print("\ninput=", " ".join(idx2src[idx] for idx in x[0]))
                    # print("expected=", " ".join(idx2tgt[idx] for idx in y[0]))
                    # print("got=", " ".join(idx2tgt[idx] for idx in _preds[0]))
                    # gs = _gs
                    # plot_alignment(_alignments[0], _gs)

            # gs, _ = sess.run([g.global_step, g.merged], {g.x: x, g.y:y})
            sv.saver.save(sess, hpl.logdir + '/model_epoch_%02d_gs_%d' % (epoch, i))
            # gs = sess.run(g.global_step)
            # sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Done")