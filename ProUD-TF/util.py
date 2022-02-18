# -*- coding: utf-8 -*-
# Renjun Hu, Feb 12, 2019

import numpy as np
import pandas as pd
import argparse
import time
from collections import defaultdict
import os
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression


def transform_item(s, item2id):
    ls = []
    for item in s.split(','):
        if len(item) == 0:
            continue
        if not item in item2id:
            item2id[item] = len(item2id)
        ls.append( item2id[item] )
    return ls


def load(args):
    """load train/valid/test data"""
    item2id = {'Placeholder': 0}
    user_feat_size = 0
    feat_train, feat_test = defaultdict(list), defaultdict(list)
    
    # process training data
    fname_train = '%strain' % args.dir
    data_train = pd.read_csv(fname_train, delimiter='\t', dtype=str, header=None).values
    # data_train line format: 0:line_id 1:label 2:poi_feat 3:user_feat 4:action_feat
    for d in data_train:
        feat_train['id'].append(int(d[0]))
        feat_train['label'].append(int(d[1]))
        feat_train['poi'].append(transform_item(d[2], item2id))
        feat_train['user'].append(transform_item(d[3], item2id))
        feat_train['action'].append(transform_item(d[4], item2id))
        user_feat_size = max(user_feat_size, len(feat_train['user'][-1]))
    
    # process test data
    fname_test = '%stest' % args.dir 
    data_test = pd.read_csv(fname_test, delimiter='\t', dtype=str, header=None).values
    for d in data_test:
        feat_test['id'].append(int(d[0]))
        feat_test['label'].append(int(d[1]))
        feat_test['poi'].append(transform_item(d[2], item2id))
        feat_test['user'].append(transform_item(d[3], item2id))
        feat_test['action'].append(transform_item(d[4], item2id))
        user_feat_size = max(user_feat_size, len(feat_test['user'][-1]))
    
    args.item_size, args.user_feat_size = len(item2id), user_feat_size
    print 'user_feat_size is %d, in %f seconds' % (user_feat_size, time.clock())
    
    # generate train/valid/test with np array
    idx = range(data_train.shape[0])
    np.random.shuffle(idx)
    split_pos = len(idx) * 9 / 10
    train = {'id': np.zeros([split_pos], dtype=np.int32),
             'poi': np.zeros([split_pos, args.poi_feat_size], dtype=np.int32),
            'user': np.zeros([split_pos, args.user_feat_size], dtype=np.int32),
            'action': np.zeros([split_pos, args.action_feat_size], dtype=np.int32),
            'label': np.zeros([split_pos], dtype=np.int32)}
    processed = 0
    for i in idx[:split_pos]:
        train['id'][processed] = feat_train['id'][i]
        train['label'][processed] = feat_train['label'][i]
        train['poi'][processed][:len(feat_train['poi'][i])] = feat_train['poi'][i]
        train['user'][processed][:len(feat_train['user'][i])] = feat_train['user'][i]
        train['action'][processed][:len(feat_train['action'][i])] = feat_train['action'][i]
        processed += 1
    valid = {'id': np.zeros([len(idx)-split_pos], dtype=np.int32),
             'poi': np.zeros([len(idx)-split_pos, args.poi_feat_size], dtype=np.int32),
            'user': np.zeros([len(idx)-split_pos, args.user_feat_size], dtype=np.int32),
            'action': np.zeros([len(idx)-split_pos, args.action_feat_size], dtype=np.int32),
            'label': np.zeros([len(idx)-split_pos], dtype=np.int32)}
    processed = 0
    for i in idx[split_pos:]:
        valid['id'][processed] = feat_train['id'][i]
        valid['label'][processed] = feat_train['label'][i]
        valid['poi'][processed][:len(feat_train['poi'][i])] = feat_train['poi'][i]
        valid['user'][processed][:len(feat_train['user'][i])] = feat_train['user'][i]
        valid['action'][processed][:len(feat_train['action'][i])] = feat_train['action'][i]
        processed += 1
    test = {'id': np.zeros([data_test.shape[0]], dtype=np.int32),
            'poi': np.zeros([data_test.shape[0], args.poi_feat_size], dtype=np.int32),
            'user': np.zeros([data_test.shape[0], args.user_feat_size], dtype=np.int32),
            'action': np.zeros([data_test.shape[0], args.action_feat_size], dtype=np.int32),
            'label': np.zeros([data_test.shape[0]], dtype=np.int32)}
    for i in range(data_test.shape[0]):
        test['id'][i] = feat_test['id'][i]
        test['label'][i] = feat_test['label'][i]
        test['poi'][i][:len(feat_test['poi'][i])] = feat_test['poi'][i]
        test['user'][i][:len(feat_test['user'][i])] = feat_test['user'][i]
        test['action'][i][:len(feat_test['action'][i])] = feat_test['action'][i]
    
    print 'loading data in %f seconds' % time.clock()
    return train, valid, test


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='../beijing/', 
                        help='directory')
    parser.add_argument('--l2_weight', type=float, default=1e-5, 
                        help='weight of l2 regularization')
    parser.add_argument('--reg_weight', type=float, default=0.03, 
                        help='importance of regularization term')
    parser.add_argument('--dropout', type=float, default=0.4, 
                        help='dropout rate')
    parser.add_argument('--max_steps', type=int, default=100000, 
                        help='maximal number of steps')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size') 
    parser.add_argument('--patience', type=int, default=10, 
                        help='number of epochs for early stopping')
    parser.add_argument('--fnn_hidden', type=int, default=16, 
                        help='size of hidden layer in FNN')
    parser.add_argument('--warmup_steps', type=int, default=0, 
                        help='warmup steps')  # automatically determined in running time
    parser.add_argument('--item_size', type=int, default=0, 
                        help='number of items to embed')  # automatically determined in running time
    parser.add_argument('--poi_feat_size', type=int, default=3, 
                        help='number of POI features')    # ID & tag & location
    parser.add_argument('--user_feat_size', type=int, default=0, 
                        help='upper bound number of user features')  # automatically determined in running time
    parser.add_argument('--action_feat_size', type=int, default=2, 
                        help='number of action features')  # time & location
    parser.add_argument('--dim', type=int, default=64, 
                        help='dimension of model')
    parser.add_argument('--epsilon', type=float, default=1e-6, 
                        help='epsilon of Adam optimizer')
    parser.add_argument('--cuda_devices', type=str, default='7', 
                        help='CUDA_VISIBLE_DEVICES')
    return parser.parse_args()


def metric_results(pred, label, thre=None):
    """pred has already applied the sigmoid function, label values are either 0 or 1"""
    #ls = []
    #for i in range(100):
    #    ls.append('%d %f' % (label[i], pred[i]))
    #print '\t'.join(ls)
    if thre is not None:
        thre_best = thre
    else:
        ls = [ (pred[i], label[i]) for i in range(pred.shape[0]) ]
        ls.sort(key=lambda x: x[0], reverse=True)
        num_pos = np.sum(label)
        f1_best, thre_best, num_acc = -1., 0., 0
        for i in range(pred.shape[0]):
            if ls[i][1] == 1:
                num_acc += 1
            cp, cr = float(num_acc) / (i + 1), float(num_acc) / num_pos
            if cp + cr > 0. and 2. * cp * cr / (cp + cr) > f1_best:
                f1_best = 2. * cp * cr / (cp + cr)
                thre_best = ls[i][0]
    #print f1_best
    pred_label = np.greater_equal(pred, thre_best)    
    #mae = np.mean(np.abs(pred - label))
    #rmse = np.sqrt(np.mean(np.square(pred - label)))
    prec, recall, f1, _ = precision_recall_fscore_support(label, pred_label, average='binary')
    auc = roc_auc_score(label, pred)
    return prec, recall, f1, auc, thre_best


def logistic_regression(X, y, model=None):
    """train a logistic regression model"""
    if model is None:
        best = None
        best_f1 = -1.
        for c in [1e-3, 1e-2, 1e-1, 1., 10, 100, 1000]:  # 1e-4, 1e-3, 1e-2, 1e-1, 1., 10, 100, 1000, 10000
            model = LogisticRegression(C=c, solver='lbfgs', class_weight='balanced').fit(X, y)
            pred_label = model.predict(X)
            #print y[:100]
            #print pred_label[:100]
            prec, recall, f1, _ = precision_recall_fscore_support(y, pred_label, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best = model
        model = best
    pred_label = model.predict(X)
    pred = model.predict_proba(X)[:, 1]
    #if model is not None:
    #    print pred[:100]
    
    prec, recall, f1, _ = precision_recall_fscore_support(y, pred_label, average='binary')
    auc = roc_auc_score(y, pred)
    return prec, recall, f1, auc, model
    