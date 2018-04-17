#!/usr/bin/env python
import sys, torch, os, numpy
from .dialog_tracker import Dialog_Tracker
from ailab.utils import Config, setLogger
import torch.optim as optim

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Supervised:
    def __init__(self, cfg_fn, reader_cls):
        '''
            Policy Gradiant for end2end chatbot

            Input:
                - cfg_fn: config file location
                - reader_class: class for reader
        '''
        self.cfg = Config(cfg_fn)
        if not torch.cuda.is_available(): 
            self.cfg.model.use_gpu = 0
        self.logger = setLogger(self.cfg.logger)
        
        self.__init_reader(reader_cls)
        self.__init_tracker()


    def __init_reader(self, reader_cls):
        #reader
        self.reader = reader_cls(self.cfg)
        self.reader.build_responses() #build response template index, will read response template and create entity need for each response template every time, but not search index.
        self.reader.responses.build_mask()


    def __init_tracker(self):
        '''tracker'''
        self.tracker =  Dialog_Tracker(self.cfg.model, self.reader.vocab, len(self.reader))
        self.tracker.network()
        
        #use pretrained word2vec
        self.tracker.encoder.embedding.weight.data = torch.FloatTensor(self.tracker.vocab.dense_vectors())

        if self.cfg.model.use_gpu:
            self.tracker.cuda(self.cfg.model.use_gpu-1)

        #checkpoint
        if os.path.exists(self.cfg.model['saved_model']):
            if self.cfg.model.use_gpu:
                checkpoint = torch.load(self.cfg.model['saved_model'])
            else:
                checkpoint = torch.load(self.cfg.model['saved_model'], map_location={'cuda:0': 'cpu'})
            self.tracker.load_state_dict(checkpoint['model'])
        
        self.loss_function = torch.nn.NLLLoss()
        self.optimizer = optim.Adam(self.tracker.parameters(), lr=self.cfg.model.learning_rate, weight_decay=self.cfg.model.weight_decay)


    def read(self, filepath):
        '''
            read train data

            Input:
                - filepath: file location
        '''
        self.reader.read(filepath)


    def train(self):
        for epoch, d in enumerate(self.reader):
            self.tracker.zero_grad()
            y_prob = torch.log(self.tracker(d))
        
            responses = d['response']

            loss = self.loss_function(y_prob, responses)
        
            #precision
            _, y_pred = torch.max(y_prob.data, 1)
            precision = y_pred.eq(responses.data).sum()/responses.numel()
            
            self.logger.info('{} {} {} {}'.format(epoch, self.cfg.model.epochs, loss.data[0], precision))
        
            loss.backward()
            self.optimizer.step()

            #save
            if epoch > 0 and epoch%1000 == 0: 
                model_state_dict = self.tracker.state_dict()
                optimizer_state_dict = self.optimizer.state_dict()
                checkpoint = {
                    'model': model_state_dict,
                    'optimizer': optimizer_state_dict,
                }
                torch.save(checkpoint, self.cfg.model['saved_model'])


