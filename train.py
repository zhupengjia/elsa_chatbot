#!/usr/bin/env python
import sys, torch
from src.reader import Reader_Dialog
from src.model.dialog_tracker import Dialog_Tracker
from nlptools.utils import Config, setLogger
import torch.optim as optim

config = Config('config/hr.yml')
if not torch.cuda.is_available(): config.model.use_gpu = 0

logger = segLogger(config.logger)

data = Reader_Dialog(config)
data.build_responses()

locdir = '/home/pzhu/data/accenture/HR/history'
data.read(locdir)

tracker = Dialog_Tracker(config.model, data.vocab, len(data))
tracker.network()

if config.model.use_gpu:
    tracker.cuda(config.model.use_gpu-1)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(tracker.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)

for epoch, d in enumerate(data):
    tracker.zero_grad()
    y_prob = tracker(d['utterance'], d['entity'], d['mask'])
    loss = loss_function(y_prob, d['response'])

    #precision
    _, y_pred = torch.max(y_prob.data, 1)
    precision = y_pred.eq(d['response'].data).sum()/d['response'].numel()
    
    logger.info('{} {} {} {}'.format(epoch, config.model.epochs, loss.data[0], precision))

    out.backward()
    optimizer.step()
    

