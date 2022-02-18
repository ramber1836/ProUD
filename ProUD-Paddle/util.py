# -*- coding: utf-8 -*-
# Renjun Hu, Feb 12, 2019

from __future__ import print_function
import numpy as np
import pandas as pd
import argparse
import time
import random
import paddle
from collections import defaultdict
import os
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

from paddle.io import Dataset

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)

class MyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.feat = data['feat']
        self.label = data['label']
    
    def __getitem__(self, idx):
        return self.feat[idx], self.label[idx]
    
    def __len__(self):
        return len(self.feat)

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
    '''load train/valid/test data'''
    train, valid, test = {}, {}, {}
    for split in ['train_data', 'valid_data', 'test_data']:
        src = '%s%s_%d' % (args.dir, split, args.copy)
        data = pd.read_csv(src, header=None, delimiter='\t', dtype=np.int32).values
        np.random.shuffle(data)
        if split == 'train_data':
            if args.neg_per_pos != -1:  # randomly selecting negative samples to meet neg/pos ratio
                pos = [idx for idx in range(data.shape[0]) if data[idx][-1] == 1]
                neg = [idx for idx in range(data.shape[0]) if data[idx][-1] == 0]
                np.random.shuffle(neg)
                neg = neg[:len(pos) * args.neg_per_pos]
                remain = pos + neg
                remain.sort()
                data = data[remain]
            train['feat'] = data[:, :-1]
            train['label'] = data[:, -1]
        elif split == 'valid_data':
            valid['feat'] = data[:, :-1]
            valid['label'] = data[:, -1]
        else:
            test['feat'] = data[:, :-1]
            test['label'] = data[:, -1]
    args.feat_size = train['feat'].shape[1]
    args.item_size = max(np.max(train['feat']), np.max(valid['feat']), np.max(test['feat'])) + 1
    print('loading data in %f seconds' % time.clock())
    return train, valid, test


def parse_arg():
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--dir', type=str, default='./nyc/', 
                        help='directory')
    parser.add_argument('--copy', type=int, default=0, 
                        help='data copy (train/valid/test split)')
    parser.add_argument('--item_size', type=int, default=0, 
                        help='number of items to embed')  # automatically determined in running time
    parser.add_argument('--feat_size', type=int, default=0, 
                            help='number of features')    # automatically determined in running time  
    # model arguments
    parser.add_argument('--likelihood_reg', type=float, default=1e-5, 
                        help='weight of likelihood L2 regularization')
    parser.add_argument('--l2_weight', type=float, default=1e-5, 
                        help='weight of l2 regularization')
    parser.add_argument('--neg_per_pos', type=int, default=-1, 
                        help='number of negative samples per positive one')
    parser.add_argument('--dim', type=int, default=64, 
                        help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0.2, 
                        help='dropout rate')
    # training arguments
    parser.add_argument('--max_epochs', type=int, default=100, 
                        help='maximal number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.7, 
                        help='decay rate of learning rate per epoch')
    parser.add_argument('--batch_size', type=int, default=512, 
                        help='batch size') 
    parser.add_argument('--patience', type=int, default=5, 
                        help='number of epochs for early stopping')
    parser.add_argument('--epsilon', type=float, default=1e-8, 
                        help='epsilon of Adam optimizer')
    parser.add_argument('--cuda_devices', type=str, default='3', 
                        help='CUDA_VISIBLE_DEVICES')
    return parser.parse_args()


def metric_results(pred, label, thre=None):
    '''pred has already applied the sigmoid function, label values are either 0 or 1'''
    if thre is not None:
        thre_best = thre
    else:
        ls = [(pred[i], label[i]) for i in range(len(pred))]
        ls = sorted(ls, key=lambda x: x[0], reverse=True)
        num_pos = np.sum(label)
        f1_best, thre_best, num_acc = -1., 0., 0
        for i in range(len(pred)):
            if ls[i][1] == 1:
                num_acc += 1
            cp, cr = float(num_acc) / (i + 1), float(num_acc) / num_pos
            if cp + cr > 0. and 2. * cp * cr / (cp + cr) > f1_best:
                f1_best = 2. * cp * cr / (cp + cr)
                thre_best = ls[i][0]
    pred_label = np.greater_equal(pred, thre_best)    
    prec, recall, f1, _ = precision_recall_fscore_support(label, pred_label, average='binary')
    auc = roc_auc_score(label, pred)
    return prec, recall, f1, auc, thre_best


def logistic_regression(X, y, model=None):
    '''train a logistic regression model'''
    if model is None:
        best = None
        best_f1 = -1.
        for c in [1e-3, 1e-2, 1e-1, 1., 10, 100, 1000]:  # 1e-4, 1e-3, 1e-2, 1e-1, 1., 10, 100, 1000, 10000
            model = LogisticRegression(C=c, solver='lbfgs', class_weight='balanced').fit(X, y)
            pred_label = model.predict(X)
            prec, recall, f1, _ = precision_recall_fscore_support(y, pred_label, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best = model
        model = best
    pred_label = model.predict(X)
    pred = model.predict_proba(X)[:, 1]
    
    prec, recall, f1, _ = precision_recall_fscore_support(y, pred_label, average='binary')
    auc = roc_auc_score(y, pred)
    return prec, recall, f1, auc, model, pred
    