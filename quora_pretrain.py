#!/usr/bin/env python3
import sys, torch, os
from nlptools.utils import Config, setLogger
import torch.nn as nn
import torch.optim as optim
from src.reader.reader_quora import Reader_Quora
from src.model.duplicate_embedding import Duplicate_Embedding

cfg = Config('config/quora.yaml')
if not torch.cuda.is_available(): cfg.use_gpu = 0
logger = setLogger(cfg)

data = Reader_Quora(cfg)
data.shuffle()

model = Duplicate_Embedding(cfg, data.vocab)
model.network()

if cfg.use_gpu:
    model.cuda(cfg.use_gpu-1)

#print(model(data[0]['question1'], data[0]['question2']))

loss_function = nn.BCELoss()

#optimizer
optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

#load checkpoint
if os.path.exists(cfg['saved_model']):
    checkpoint = torch.load(cfg['saved_model'])
    model.load_state_dict(checkpoint['model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

#train loop

for epoch in range(300):
    data.shuffle()
    N = 0
    for i, d in enumerate(data):
        model.zero_grad()
          
        y_probs = model(d['question1'], d['question2'])
        out = loss_function(y_probs, d['match'].view(-1,1))

        y_pred = (y_probs >= 0.5).squeeze(1)
        y_true = d['match'] >=0.5
        
        correct = y_pred.eq(y_true).long().sum()
        total = d['match'].numel()
      
        precision = correct.data.cpu().numpy()/total

        out.backward()
        optimizer.step()
        N += 1
        
        logger.info('{} {} {} {} {}'.format(epoch, N, len(data), out.data[0], precision[0]))

    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    checkpoint = {
        'model': model_state_dict,
        'optimizer': optimizer_state_dict,
    }
 
    torch.save(checkpoint, cfg['saved_model'])

