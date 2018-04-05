#!/usr/bin/env python
import sys, torch, os
from src.reader.reader_babi import Reader_Babi
from src.model.rule_based import Rule_Based
from src.hook.babi_gensays import Babi_GenSays
from src.model.dialog_tracker import Dialog_Tracker
from nlptools.utils import Config, setLogger
import torch.optim as optim

config = Config('config/babi.yml')
if not torch.cuda.is_available(): config.model.use_gpu = 0

logger = setLogger(config.logger)

hook = Babi_GenSays(config)
ad_chatbot = Rule_Based(config.rule_based, hook)
print(ad_chatbot.get_reply('let me do the reservation.', 1))






sys.exit()





data = Reader_Babi(config)
data.build_responses() #build response template index, will read response template and create entity need for each response template every time, but not search index.

datafile = '/home/pzhu/data/dialog/babi/dialog-babi-task5-full-dialogs-trn.txt'
data.read(datafile) #read data and predeal, will directly use cached data if existed, but will build mask every time

tracker = Dialog_Tracker(config.model, data.vocab, len(data))
tracker.network()

#use pretrained word2vec
tracker.encoder.embedding.weight.data = torch.FloatTensor(tracker.vocab.dense_vectors())

if config.model.use_gpu:
    tracker.cuda(config.model.use_gpu-1)

loss_function = torch.nn.NLLLoss()
optimizer = optim.Adam(tracker.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)


#load checkpoint
if os.path.exists(config.model['saved_model']):
    if config.model.use_gpu:
        checkpoint = torch.load(config.model['saved_model'])
    else:
        checkpoint = torch.load(config.model['saved_model'], map_location={'cuda:0': 'cpu'})
    tracker.load_state_dict(checkpoint['model'])


for epoch, d in enumerate(data):
    tracker.zero_grad()
    y_prob = tracker(d)

    responses = d['response']
    loss = loss_function(y_prob, responses)

    #precision
    _, y_pred = torch.max(y_prob.data, 1)
    precision = y_pred.eq(responses.data).sum()/responses.numel()
    
    logger.info('{} {} {} {}'.format(epoch, config.model.epochs, loss.data[0], precision))

    loss.backward()
    optimizer.step()
    if epoch > 0 and epoch%1000 == 0: 
        model_state_dict = tracker.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'optimizer': optimizer_state_dict,
        }
        torch.save(checkpoint, config.model['saved_model'])

