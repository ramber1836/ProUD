# -*- coding: utf-8 -*-
from __future__ import print_function, division
from util import parse_arg, load, metric_results, MyDataset, set_seeds
from model import MyModel
import numpy as np
import paddle
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from paddle.io import DataLoader

valid_f1_mx, valid_thre, epochs_no_gain = 0., .5, 0
np.set_printoptions(precision=2)

args = parse_arg()
set_seeds(42)

train, valid, test = load(args)
train_set, valid_set, test_set = MyDataset(train), MyDataset(valid), MyDataset(test)
train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True)
valid_loader = DataLoader(valid_set, batch_size = args.batch_size, shuffle = False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
paddle.set_device('gpu')
checkpt_file = './model_ckpt/on_cuda_%s.ckpt' % (args.cuda_devices)
for arg in vars(args):
    print(arg, '\t', getattr(args, arg))

model = MyModel(args)
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=5, gamma=args.lr_decay)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=args.l2_weight)

for epoch in range(args.max_epochs):
    losses = []
    # train
    for feat, label in train_loader:
        loss, _ = model(feat, label)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        losses.append(loss.item())
    print('Epoch %d loss: %.6f' % (epoch + 1, np.mean(losses)))
    
    # validation
    model.eval()
    pred_valid = []
    valid_label = []
    for feat, label in valid_loader:
        _, pred = model.forward(feat, label)
        pred_valid += pred.tolist()
        valid_label += label.tolist()

    prec, recall, f1, auc, thre = metric_results(pred_valid, valid_label)
    print('Epoch %d  v_Prec: %.6f  v_Recall: %.6f  v_F1: %.6f  v_AUC: %.6f  thre: %.6f' % (epoch+1, prec, recall, f1, auc, thre))
    print('Epoch %d  running time %.0f seconds' % (epoch + 1, time.clock()))

    if f1 > valid_f1_mx:
        valid_f1_mx, valid_thre = f1, thre
        paddle.save(model.state_dict(), checkpt_file)
        epochs_no_gain = 0
    else:
        epochs_no_gain += 1
        if epochs_no_gain == args.patience:
            print('Early stop model F1: %.6f' % valid_f1_mx)
            break
    
# test
checkpoint = paddle.load(checkpt_file) 
model.set_state_dict(checkpoint)
model.eval()

pred_valid = []
valid_label = []
for feat, label in valid_loader:
    _, pred = model.forward(feat, label)
    pred_valid += pred.tolist()
    valid_label += label.tolist()
_, _, _, _, valid_thre = metric_results(pred_valid, valid_label)

pred_test = []
test_label = []
for feat, label in test_loader:
    _, pred = model.forward(feat, label)
    pred_test += pred.tolist()
    test_label += label.tolist()
print('Testing Thre:', valid_thre)
prec, recall, f1, auc, _ = metric_results(pred_test, test_label, valid_thre)
print('Testing Prec:', prec)
print('Testing Recall:', recall)
print('Testing F1:', f1)
print('Testing AUC:', auc)