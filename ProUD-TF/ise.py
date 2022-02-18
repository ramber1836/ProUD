# -*- coding: utf-8 -*-
# Renjun Hu, Feb 12, 2019

from __future__ import print_function
import numpy as np
import tensorflow as tf

class ItemsetEmbedding(object):
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
        return
    
    def _build_model(self, args):
        """define model base, intermediate and predictive variables"""
        # base 
        init = tf.initializers.random_uniform(-(6.0/args.dim)**0.5, (6.0/args.dim)**0.5)
        self.base_embed = tf.get_variable("base_embed", [args.item_size, args.dim], dtype=tf.float32, initializer=init)
        
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
        self.length = tf.norm(poi_embed+user_embed+action_embed, ord='euclidean', axis=-1)
        self.pred = poi_embed + user_embed + action_embed
        return
    
    def _build_train(self, args):
        label_ft = tf.to_float(self.label) 
        obj = -label_ft * tf.log(tf.tanh(.5 * self.length)) - (1.-label_ft) * tf.log(tf.tanh(.5 * (1./self.length)))
        self.loss = tf.reduce_mean(obj)
        
        self.global_steps = tf.train.get_or_create_global_step()
        step = tf.to_float(self.global_steps)
        warmup_steps = tf.to_float(args.warmup_steps)
        multiplier = 3. * args.dim ** -0.5
        lr = multiplier * tf.minimum(step * (warmup_steps ** -1.5), step ** -0.5)
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.epsilon)\
            .minimize(self.loss, global_step=self.global_steps)
        return
    
    def train(self, sess, feed_dict):
        _, loss, pred = sess.run([self.train_op, self.loss, self.pred], feed_dict)
        return loss, pred
    
    def get_pred(self, sess, feed_dict):
        pred = sess.run(self.pred, feed_dict)
        #return 1. / (1. + np.exp(-pred))  # applying sigmoid function
        return pred
    
    def list_variables(sess):
        return