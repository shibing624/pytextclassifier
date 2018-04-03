# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Document classification model

import numpy as np
import tensorflow as tf

import config
from evaluate import simple_evaluate
from layers.cnn_layer import CNN
from layers.dense_layer import SoftmaxDense
from layers.emb_layer import Embedding
from tensor_utils import zero_nil_slot


class Model(object):
    def __init__(self, max_len, word_emb, pos_emb, label_vocab=None):
        """
        Init model
        :param max_len:
        :param word_emb:
        :param pos_emb:
        :param label_vocab:
        """
        self._label_vocab = label_vocab
        self._label_vocab_rev = dict()
        for key in self._label_vocab:
            value = self._label_vocab[key]
            self._label_vocab_rev[value] = key

        # input placeholders
        self.input_sentence_ph = tf.placeholder(
            tf.int32, shape=(None, max_len), name='input_sentence_ph'
        )
        self.input_pos_ph = tf.placeholder(tf.int32, shape=(None, max_len), name='input_pos_ph')
        self.label_ph = tf.placeholder(tf.int32, shape=(None,), name='label_ph')
        self.keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob')
        self.word_keep_prob_ph = tf.placeholder(tf.float32, name="word_keep_prob")
        self.pos_keep_prob_ph = tf.placeholder(tf.float32, name='pos_keep_prob')

        # emb
        self.nil_vars = set()
        word_embed_layer = Embedding(params=word_emb, ids=self.input_sentence_ph,
                                     keep_prob=self.word_keep_prob_ph, name='word_embed_layer')
        pos_emb_layer = Embedding(params=pos_emb, ids=self.input_pos_ph,
                                  keep_prob=self.pos_keep_prob_ph, name='pos_embed_layer')
        self.nil_vars.add(word_embed_layer.params.name)
        self.nil_vars.add(pos_emb_layer.params.name)

        # sentence representation
        sentence_input = tf.concat(values=[word_embed_layer.output, pos_emb_layer.output], axis=2)

        # sentence conv
        conv_layer = CNN(input_data=sentence_input, filter_length=3,
                         nb_filter=1000, activation='relu', name='conv_layer')

        # dense layer
        dense_input_drop = tf.nn.dropout(conv_layer.output, self.keep_prob_ph)
        self.dense_layer = SoftmaxDense(input_data=dense_input_drop, input_dim=conv_layer.output_dim,
                                        output_dim=config.nb_labels, name='output_layer')

        self.loss = self.dense_layer.loss(self.label_ph) + \
                    0.001 * tf.nn.l2_loss(self.dense_layer.weights)
        optimizer = tf.train.AdamOptimizer()
        grads_and_vars = optimizer.compute_gradients(self.loss)
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self.nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # train op
        self.train_op = optimizer.apply_gradients(nil_grads_and_vars, name='train_op', global_step=global_step)

        # pred op
        self.pred_op = self.dense_layer.get_pred_y()

        # summary
        gpu_options = tf.GPUOptions(visible_device_list='0', allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # init model
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()

    def fit(self, sentence_train, pos_train, label_train,
            sentence_dev=None, pos_dev=None, label_dev=None,
            sentence_test=None, pos_test=None, label_test=None,
            batch_size=64, nb_epoch=40, keep_prob=1.0,
            word_keep_prob=1.0, pos_keep_prob=1.0, seed=137):
        """
        Fit model
        """
        self.nb_epoch_scores = []  # save nb_epoch F1 score
        nb_train = int(label_train.shape[0] / batch_size) + 1
        for step in range(nb_epoch):
            print('Epoch %d / %d:' % (step + 1, nb_epoch))
            # shuffle
            np.random.seed(seed)
            np.random.shuffle(sentence_train)
            np.random.seed(seed)
            np.random.shuffle(pos_train)
            np.random.seed(seed)
            np.random.shuffle(label_train)

            # train
            total_loss = 0
            for i in range(nb_train):
                sentence_feed = sentence_train[i * batch_size:(i + 1) * batch_size]
                pos_feed = pos_train[i * batch_size:(i + 1) * batch_size]
                label_feed = label_train[i * batch_size:(i + 1) * batch_size]
                feed_dict = {
                    self.input_sentence_ph: sentence_feed,
                    self.input_pos_ph: pos_feed,
                    self.label_ph: label_feed,
                    self.keep_prob_ph: keep_prob,
                    self.word_keep_prob_ph: word_keep_prob,
                    self.pos_keep_prob_ph: pos_keep_prob,
                }
                _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                total_loss += loss_value
            total_loss /= float(nb_train)

            # evaluation
            p_train, r_train, f_train = self.eval(sentence_train, pos_train, label_train)
            p_dev, r_dev, f_dev = self.eval(sentence_dev, pos_dev, label_dev)
            print("p_train:%f, p_dev:%f" % (p_train, p_dev))
            pred_labels = self.predict(sentence_test, pos_test)
            with open(config.model_save_dir + '/epoch_%d.csv' % (step + 1), 'w', encoding='utf-8') as f:
                for num, label in enumerate(pred_labels):
                    f.write('%d,%s\n' % (num + 1, self._label_vocab_rev[label]))
            self.nb_epoch_scores.append([p_dev, r_dev, f_dev])
            print('\tloss=%f, train f=%f, dev f=%f' % (total_loss, f_train, f_dev))

    def save(self, model_path):
        self.saver.save(self.sess, model_path)

    def predict(self, data_sentence, data_pos, batch_size=50):
        """
        Predict data result
        :param data_sentence:
        :param data_pos:
        :param batch_size:
        :return:
        """
        pred_labels = []
        nb_test = int(data_sentence.shape[0] / batch_size) + 1
        for i in range(nb_test):
            sentence_feed = data_sentence[i * batch_size:(i + 1) * batch_size]
            pos_feed = data_pos[i * batch_size:(i + 1) * batch_size]
            feed_dict = {
                self.input_sentence_ph: sentence_feed,
                self.input_pos_ph: pos_feed,
                self.keep_prob_ph: 1.0,
                self.word_keep_prob_ph: 1.0,
                self.pos_keep_prob_ph: 1.0,
            }
            pred_temp = self.sess.run(self.pred_op, feed_dict=feed_dict)
            pred_labels += list(pred_temp)
        return pred_labels

    def eval(self, data_sentence, data_pos, data_label, batch_size=64):
        """
        Evaluate data result
        :param data_sentence:
        :param data_pos:
        :param data_label:
        :param batch_size:
        :return:
        """
        pred_labels = []
        nb_dev = int(len(data_label) / batch_size) + 1
        for i in range(nb_dev):
            sentence_feed = data_sentence[i * batch_size:(i + 1) * batch_size]
            pos_feed = data_pos[i * batch_size:(i + 1) * batch_size]
            label_feed = data_label[i * batch_size:(i + 1) * batch_size]
            feed_dict = {
                self.input_sentence_ph: sentence_feed,
                self.input_pos_ph: pos_feed,
                self.label_ph: label_feed,
                self.keep_prob_ph: 1.0,
                self.word_keep_prob_ph: 1.0,
                self.pos_keep_prob_ph: 1.0
            }
            pred_temp = self.sess.run(self.pred_op, feed_dict=feed_dict)
            pred_labels += list(pred_temp)
        true_labels = data_label[:len(pred_labels)]
        p, r, f = simple_evaluate(true_labels, pred_labels)
        return p, r, f

    def clear_model(self):
        tf.reset_default_graph()
        self.sess.close()

    def get_best_score(self):
        """
        Compute the best score of model
        """
        nb_epoch, best_score = -1, None
        for i in range(len(self.nb_epoch_scores)):
            if not best_score or self.nb_epoch_scores[i][-1] > best_score[-1]:
                best_score = self.nb_epoch_scores[i]
                nb_epoch = i
        return best_score, nb_epoch
