# -*- coding: utf-8 -*-
# Renjun Hu, Feb 12, 2019

from util import parse_arg, load, metric_results
from model import MyModel
import numpy as np
import tensorflow as tf
import os
import time


def get_feed_dict(model, data, start, end, dropout=0.):
    feed_dict = {model.label: data['label'][start:end],
                model.poi_feat: data['poi'][start:end],
                model.user_feat: data['user'][start:end],
                model.action_feat: data['action'][start:end],
                model.keep_prob: 1. - dropout}
    return feed_dict


args = parse_arg()
train, valid, test = load(args)
print 'train/valid/test size: %d/%d/%d' % (train['label'].shape[0], valid['label'].shape[0], test['label'].shape[0])
args.warmup_steps = train['label'].shape[0]/args.batch_size
for arg in vars(args):
    print arg, '\t', getattr(args, arg) 

model = MyModel(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
saver = tf.train.Saver()
valid_f1_mx = 0.
valid_thre = .5
epochs_no_gain = 0
checkpt_file = './model_ckpt/on_cuda_%s.ckpt' % (args.cuda_devices)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    #model.list_variables(sess)
    
    start_list = range(0, train['label'].shape[0], args.batch_size)
    base_losses, reg_losses, l2_losses, losses = [], [], [], []
    for step in range(args.max_steps):
        start = start_list[step % len(start_list)]
        base_loss, reg_loss, l2_loss = model.train(
            sess, get_feed_dict(model, train, start, start+args.batch_size, args.dropout))
        base_losses.append(base_loss)
        reg_losses.append(reg_loss)
        l2_losses.append(l2_loss)
        losses.append(base_loss+reg_loss+l2_loss)
        
        if (step + 1) % len(start_list) == 0:
            np.random.shuffle(start_list)
        if (step + 1) % 200 == 0:
            print('steps %d    base_loss: %.6f    reg_loss: %.6f    l2_loss: %.6f    loss: %.6f' % 
                  (step+1, np.mean(base_losses), np.mean(reg_losses), np.mean(l2_losses), np.mean(losses)))
            del base_losses[:]
            del reg_losses[:]
            del l2_losses[:]
            del losses[:]
            # validation
            pred_valid = []
            for start in range(0, valid['label'].shape[0], args.batch_size):
                pred = model.get_pred(
                    sess, get_feed_dict(model, valid, start, start+args.batch_size))
                pred_valid.append(pred)
            pred_valid = np.concatenate(pred_valid)
            prec, recall, f1, auc, thre = metric_results(pred_valid, valid['label'])
            print 'steps %d    v_Prec: %.6f  v_Recall: %.6f  v_F1: %.6f  v_AUC: %.6f' % (step+1, prec, recall, f1, auc)
            #print 'steps %d    thre: %.6f' % (step+1, thre)
            print 'steps %d    running time %.0f seconds' % (step+1, time.clock())
            print
            if f1 > valid_f1_mx:
                valid_f1_mx = f1
                valid_thre = thre
                saver.save(sess, checkpt_file)
                epochs_no_gain = 0
            else:
                epochs_no_gain += 1
                if epochs_no_gain == args.patience:
                    print 'Early stop model F1: %.6f' % valid_f1_mx 
                    break
            # if step + 1 == 200:
            #     break

    saver.restore(sess, checkpt_file)
    # testing
    pred_test = []
    for start in range(0, test['label'].shape[0], args.batch_size):
        pred = model.get_pred(
            sess, get_feed_dict(model, test, start, start+args.batch_size))
        pred_test.append(pred)
    pred_test = np.concatenate(pred_test)
    print 'testing thre:', valid_thre
    prec, recall, f1, auc, _ = metric_results(pred_test, test['label'], valid_thre)
    print 'testing Prec:', prec
    print 'testing Recall:', recall
    print 'testing F1:', f1
    print 'testing AUC:', auc
    print 'overall running time %.0f seconds' % (time.clock())
    print metric_results(pred_test, test['label'])
    print
    
    # get interpretable results
    # model.list_variables(sess)
    with open('../beijing/cos_train', 'w') as fout:
        for start in range(0, train['label'].shape[0], args.batch_size):
            itp = model.get_interpretable(
                sess, get_feed_dict(model, train, start, start+args.batch_size))
            for i in range(start, start+args.batch_size):
                if i >= train['id'].shape[0]:
                    break
                fout.write('%d\t%f\t%f\t%f\n' % (train['id'][i], itp[i-start][0], itp[i-start][1], itp[i-start][2]))
        for start in range(0, valid['label'].shape[0], args.batch_size):
            itp = model.get_interpretable(
                sess, get_feed_dict(model, valid, start, start+args.batch_size))
            for i in range(start, start+args.batch_size):
                if i >= valid['id'].shape[0]:
                    break
                fout.write('%d\t%f\t%f\t%f\n' % (valid['id'][i], itp[i-start][0], itp[i-start][1], itp[i-start][2]))
    with open('../beijing/cos_test', 'w') as fout:
        for start in range(0, test['label'].shape[0], args.batch_size):
            itp = model.get_interpretable(
                sess, get_feed_dict(model, test, start, start+args.batch_size))
            for i in range(start, start+args.batch_size):
                if i >= test['id'].shape[0]:
                    break
                fout.write('%d\t%f\t%f\t%f\n' % (test['id'][i], itp[i-start][0], itp[i-start][1], itp[i-start][2]))