# -*- coding: utf-8 -*-
# Renjun Hu, Feb 12, 2019

from __future__ import print_function
import numpy as np
import tensorflow as tf

class MyModel(object):
    def __init__(self, args):
        self._build_inputs(args)
        self._build_model(args)
        self._build_train(args)
        return
    
    def _build_inputs(self, args):
        """define feed_dict variables"""
        self.poi_feat = tf.placeholder(dtype=tf.int32, shape=[None, args.poi_feat_size], name="poi_feat")
        self.user_feat = tf.placeholder(dtype=tf.int32, shape=[None, args.user_feat_size], name="user_feat")
        self.action_feat = tf.placeholder(dtype=tf.int32, shape=[None, args.action_feat_size], name="action_feat")
        self.label = tf.placeholder(dtype=tf.int32, shape=[None], name="label")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        return
    
    def _build_model(self, args):
        """define model base, intermediate and predictive variables"""
        # base 
        init = tf.initializers.random_uniform(-(6.0/args.dim)**0.5, (6.0/args.dim)**0.5)
        self.base_embed = tf.get_variable("base_embed", [args.item_size, args.dim], dtype=tf.float32, initializer=init)
        self.ltb = tf.get_variable("length_trans_bias", initializer=tf.constant_initializer(0.), 
                                   dtype=tf.float32, shape=[1, 3], trainable=True)
        
        # intermediate: feature embedding lookup
        poi_lookup = tf.nn.embedding_lookup(self.base_embed, self.poi_feat)
        user_lookup = tf.nn.embedding_lookup(self.base_embed, self.user_feat)
        action_lookup = tf.nn.embedding_lookup(self.base_embed, self.action_feat)
        
        # intermediate: feature embedding
        poi_embed = tf.reduce_sum(poi_lookup, axis=-2)             # [B, D]
        user_mul = tf.to_float(tf.not_equal(self.user_feat, 0))    # [B, user_feat_size]
        user_mul = tf.expand_dims(user_mul, axis=-1)               # expand to [B, user_feat_size, 1]
        user_embed = tf.reduce_sum(user_lookup*user_mul, axis=-2)  # [B, D]
        action_embed = tf.reduce_sum(action_lookup, axis=-2)       # [B, D]
        
        # predictive, [B]
        len_pu = tf.norm(poi_embed+user_embed, ord='euclidean', axis=-1, keep_dims=True)
        len_pa = tf.norm(poi_embed+action_embed, ord='euclidean', axis=-1, keep_dims=True)
        len_ua = tf.norm(user_embed+action_embed, ord='euclidean', axis=-1, keep_dims=True)
        fnn_feat = tf.concat([len_pu, len_pa, len_ua], axis=-1)
        #self.interpretable = fnn_feat
        #fnn_feat = tf.log(fnn_feat)
        fnn_feat -= self.ltb
        fnn_feat = tf.layers.dense(fnn_feat, units=args.fnn_hidden, 
                                   activation=tf.nn.tanh, use_bias=True, 
                                   name='fnn_layer_1', reuse=tf.AUTO_REUSE,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2_weight))
        fnn_feat = tf.nn.dropout(fnn_feat, keep_prob=self.keep_prob)
        fnn_feat = tf.layers.dense(fnn_feat, units=1, 
                                   activation=None, use_bias=False, 
                                   name='fnn_layer_2', reuse=tf.AUTO_REUSE,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2_weight))
        self.pred = tf.squeeze(fnn_feat)
        # regularization, all of size [B]
        len_poi = tf.norm(poi_embed, ord='euclidean', axis=-1)
        len_user = tf.norm(user_embed, ord='euclidean', axis=-1)
        len_action = tf.norm(action_embed, ord='euclidean', axis=-1)
        self.reg_pu = -tf.reduce_sum(poi_embed*user_embed, axis=-1) / (len_poi * len_user) + 1.
        self.reg_pa = -tf.reduce_sum(poi_embed*action_embed, axis=-1) / (len_poi * len_action) + 1.
        self.reg_ua = -tf.reduce_sum(user_embed*action_embed, axis=-1) / (len_user * len_action) + 1.
        cos_pu = tf.expand_dims(-self.reg_pu+1., axis=-1)
        cos_pa = tf.expand_dims(-self.reg_pa+1., axis=-1)
        cos_ua = tf.expand_dims(-self.reg_ua+1., axis=-1)
        self.interpretable = tf.concat([cos_pu, cos_pa, cos_ua], axis=-1)
        return
    
    def _build_train(self, args):
        label_ft = tf.to_float(self.label) 
        self.base_loss = tf.reduce_mean(-tf.log(tf.sigmoid((2.*label_ft-1.) * self.pred)))
        self.reg_loss = label_ft * (self.reg_pu + self.reg_pa + self.reg_ua)
        self.reg_loss += (1. - label_ft) * (2. - self.reg_pu + 2. - self.reg_pa)
        self.reg_loss = args.reg_weight * tf.reduce_mean(self.reg_loss)
        reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.l2_loss = tf.add_n(reg_set)
        all_item = tf.concat([self.poi_feat, self.user_feat, self.action_feat], axis=-1)
        unique_item, _ = tf.unique(tf.reshape(all_item, [-1]))
        self.l2_loss += args.l2_weight * tf.nn.l2_loss(tf.nn.embedding_lookup(self.base_embed, unique_item))
        self.loss = self.base_loss + self.reg_loss + self.l2_loss
        
        self.global_steps = tf.train.get_or_create_global_step()
        step = tf.to_float(self.global_steps)
        warmup_steps = tf.to_float(args.warmup_steps)
        multiplier = 3. * args.dim ** -0.5
        lr = multiplier * tf.minimum(step * (warmup_steps ** -1.5), step ** -0.5)
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.epsilon)\
            .minimize(self.loss, global_step=self.global_steps)
        return
    
    def train(self, sess, feed_dict):
        _, base_loss, reg_loss, l2_loss = sess.run(
            [self.train_op, self.base_loss, self.reg_loss, self.l2_loss], feed_dict)
        return base_loss, reg_loss, l2_loss
    
    def get_pred(self, sess, feed_dict):
        pred = sess.run(self.pred, feed_dict)
        return 1. / (1. + np.exp(-pred))  # applying sigmoid function
    
    def get_interpretable(self, sess, feed_dict):
        return sess.run(self.interpretable, feed_dict)
    
    def list_variables(self, sess):
        #variable_names = [v.name for v in tf.trainable_variables()]
        variable_names = ['length_trans_bias:0', 'fnn_layer_1/kernel:0', 'fnn_layer_1/bias:0', 'fnn_layer_2/kernel:0']
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable:", k, "Shape:", v.shape)
            print(v)
        return