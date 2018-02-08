#!/usr/bin/env python3
import sys, torch, os
from ailab.utils import Config, setLogger
import torch.nn as nn
import torch.optim as optim
from src.reader.quora_reader import QuoraReader
from src.model.dulicate_embedding import Duplicate_Embedding

use_gpu = 0 if torch.cuda.is_available() else 0
cfg = Config('config/quora.yaml')
logger = setLogger(cfg)


data = QuoraReader(cfg, use_gpu)
data.shuffle()

model = Duplicate_Embedding(cfg, data.vocab)
model.network()
if use_gpu:
    model.cuda(use_gpu-1)

loss_function = nn.BCELoss()

#model(data[0])

#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)


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
          
        y_probs = model(d)
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

