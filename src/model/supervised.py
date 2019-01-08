#!/usr/bin/env python
import sys, torch, os, numpy
from .dialog_tracker import Dialog_Tracker
from nlptools.utils import Config, setLogger
from ..module.entity_dict import Entity_Dict
from nlptools.text import Vocab
from nlptools.text.ner import NER
import torch.optim as optim

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Supervised:
    def __init__(self, reader, max_entity_types, kernel_num=5, kernel_size=16,  epochs=1000, weight_decay=0, learning_rate=0.001, saved_model="model.pt", dropout=0.2, logger=None):
        '''
            Policy Gradiant for end2end chatbot

            Input:
                - reader: reader instance, see ..module.reader.*.py
                - max_entity_types: int, number of entity types 
                - kernel_num: int
                - kernel_size: int
                - epochs: int
                - weight_decay: int, default is 0
                - learning_rate: float, default is 0.001
                - saved_model: str, default is "model.pt"
                - dropout: float, default is 0.2
                - logger: logger instance ,default is None
        '''
        self.reader = reader

        self.saved_model = saved_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.logger = logger 

        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.max_entity_types = max_entity_types
        self.dropout = dropout

        self.__init_reader(reader_cls)
        self.__init_tracker()


    @classmethod
    def build(cls, cfgfile): 
        '''
            construct model from config

            Input:
                - cfgfile: str, config file path
        '''
        config = Config(cfg)
        logger = setLogger(config.logger)
        vocab = Vocab(**config.vocab)
        ner = NER(**config.ner)
        embedding = Embedding(**config.embedding)

        entity_dict = Entity_Dict(vocab, **config.entity_dict) 
        
        reader =  


        model = cls(reader=reader, **config.model)

        #build reader


        pass 


    def __init_reader(self, reader_cls):
        '''
            construct reader
            
            Input:
                - reader_cls:
                    class for reader
        '''
        self.reader = reader_cls(self.vocab, self.ner, self.embedding, self.entity_dict, self.hook, self.response_cache, self.batch_size, self.logger)
        self.reader.build_responses() #build response template index, will read response template and create entity need for each response template every time, but not search index.
        self.reader.responses.build_mask()


    def __init_tracker(self):
        '''tracker'''
        self.tracker =  Dialog_Tracker(self.reader.vocab, len(self.reader), self.kernel_num, self.kernel_size, self.max_entity_types, self.dropout)
        self.tracker.network()
        
        #use pretrained word2vec
        self.tracker.encoder.embedding.weight.data = torch.FloatTensor(self.tracker.vocab.dense_vectors())

        self.tracker.to(self.device)

        #checkpoint
        if os.path.exists(self.saved_model):
            checkpoint = torch.load(self.saved_model, map_location={'cuda:0': self.device.type})
            self.tracker.load_state_dict(checkpoint['model'])
        
        self.loss_function = torch.nn.NLLLoss()
        self.optimizer = optim.Adam(self.tracker.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


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
            
            if self.ogger: self.logger.info('{} {} {} {}'.format(epoch, self.epochs, loss.data[0], precision))
        
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
                torch.save(checkpoint, self.saved_model)


