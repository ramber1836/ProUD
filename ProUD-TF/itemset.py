# -*- coding: utf-8 -*-
# Renjun Hu, Feb 12, 2019

from util import parse_arg, load, logistic_regression
from ise import ItemsetEmbedding
import numpy as np
import tensorflow as tf
import os
import time


def get_feed_dict(model, data, start, end):
    feed_dict = {model.label: data['label'][start:end],
                model.poi_feat: data['poi'][start:end],
                model.user_feat: data['user'][start:end],
                model.action_feat: data['action'][start:end]}
    return feed_dict


start_time = time.time()
args = parse_arg()
train, valid, test = load(args)
print 'train/valid/test size: %d/%d/%d' % (train['label'].shape[0], valid['label'].shape[0], test['label'].shape[0])
#args.warmup_steps = train['label'].shape[0]/args.batch_size
args.warmup_steps = 4000
for arg in vars(args):
    print arg, '\t', getattr(args, arg) 

model = ItemsetEmbedding(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
saver = tf.train.Saver()
valid_f1_mx = 0.
valid_model = None
epochs_no_gain = 0
checkpt_file = './model_ckpt/on_cuda_%s.ckpt' % (args.cuda_devices)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    #model.list_variables(sess)
    
    start_list = range(0, train['label'].shape[0], args.batch_size)
    
    for epoch in range(1, args.max_steps):
        losses, pred_train, label_train = [], [], []
        np.random.shuffle(start_list)
        for start in start_list:
            loss, pred = model.train(sess, get_feed_dict(model, train, start, start+args.batch_size))
            losses.append(loss)
            pred_train.append(pred)
            label_train.append(train['label'][start:start+args.batch_size])
            
        # train preference
        print('epoch %d    loss: %.6f' % (epoch, np.mean(losses)))
        prec, recall, f1, auc, lr_model = logistic_regression(np.concatenate(pred_train), np.concatenate(label_train))
        print 'epoch %d    t_Prec: %.6f  t_Recall: %.6f  t_F1: %.6f  t_AUC: %.6f' % (epoch, prec, recall, f1, auc)
        
        # validation
        pred_valid = []
        for start in range(0, valid['label'].shape[0], args.batch_size):
            pred = model.get_pred(
                sess, get_feed_dict(model, valid, start, start+args.batch_size))
            pred_valid.append(pred)
        prec, recall, f1, auc, lr_model = logistic_regression(np.concatenate(pred_valid), valid['label'], lr_model)
        print 'epoch %d    v_Prec: %.6f  v_Recall: %.6f  v_F1: %.6f  v_AUC: %.6f' % (epoch, prec, recall, f1, auc)
        #print 'steps %d    thre: %.6f' % (step+1, thre)
        print 'epoch %d    running time %.0f seconds' % (epoch, time.time()-start_time)
        print
        if f1 > valid_f1_mx:
            valid_f1_mx = f1
            valid_model = lr_model
            saver.save(sess, checkpt_file)
            epochs_no_gain = 0
        else:
            epochs_no_gain += 1
            if epochs_no_gain == args.patience:
                print 'Early stop model F1: %.6f' % valid_f1_mx 
                break
        #if step + 1 == 10000:
        #    break

    saver.restore(sess, checkpt_file)
    # testing
    pred_test = []
    for start in range(0, test['label'].shape[0], args.batch_size):
        pred = model.get_pred(
            sess, get_feed_dict(model, test, start, start+args.batch_size))
        pred_test.append(pred)
    prec, recall, f1, auc, _ = logistic_regression(np.concatenate(pred_test), test['label'], valid_model)
    print 'testing Prec:', prec
    print 'testing Recall:', recall
    print 'testing F1:', f1
    print 'testing AUC:', auc
    print 'overall running time %.0f seconds' % (time.time()-start_time)