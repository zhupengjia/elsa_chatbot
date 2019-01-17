#!/usr/bin/env python
import sys, torch, os, numpy
from .dialog_tracker import Dialog_Tracker
from nlptools.utils import Config, setLogger
from ..module.entity_dict import Entity_Dict
from nlptools.text import Embedding
from nlptools.text.tokenizer import Tokenizer_BERT
from nlptools.text.ner import NER
import torch.optim as optim

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Supervised:
    def __init__(self, reader, bert_model_name, max_entity_types, fc_response1=5, fc_response2=4, epochs=1000, weight_decay=0, learning_rate=0.001, saved_model="model.pt", dropout=0.2, device=None, logger=None):
        '''
            Policy Gradiant for end2end chatbot

            Input:
                - bert_model_name: bert model file location or one of the supported model name
                - reader: reader instance, see ..module.reader.*.py
                - max_entity_types: int, number of entity types 
                - epochs: int
                - weight_decay: int, default is 0
                - learning_rate: float, default is 0.001
                - saved_model: str, default is "model.pt"
                - dropout: float, default is 0.2
                - device: instance of torch.device, default is cpu device
                - logger: logger instance ,default is None
        '''
        self.reader = reader
        self.bert_model_name = bert_model_name
        self.saved_model = saved_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device
        self.logger = logger 

        self.max_entity_types = max_entity_types
        self.fc_response1 = fc_response1
        self.fc_response2 = fc_response2
        self.dropout = dropout

        self.__init_tracker()


    @classmethod
    def build(cls, config, reader_cls, hook): 
        '''
            construct model from config

            Input:
                - config: configure dictionary
                - reader_cls: class for reader
                - hook: hook instance, please check src/hook/babi_gensays.py for example
        '''
        logger = setLogger(**config.logger)
        ner = NER(**config.ner)
        tokenizer = Tokenizer_BERT(**config.tokenizer) 
        
        device = torch.device("cuda:0" if config.model.use_gpu and torch.cuda.is_available() else "cpu")

        entity_dict = Entity_Dict(**config.entity_dict) 
       
        #reader
        reader = reader_cls(tokenizer=tokenizer, ner=ner, entity_dict=entity_dict, hook=hook, max_entity_types=config.entity_dict.max_entity_types, max_seq_len=config.model.max_seq_len, epochs=config.model.epochs, batch_size=config.model.batch_size, device=device, logger=logger)
        reader.build_responses(config.response_template) #build response template index, will read response template and create entity need for each response template every time, but not search index.
        reader.response_dict.build_mask()

        return cls(reader=reader, bert_model_name=config.tokenizer.bert_model_name, max_entity_types=config.entity_dict.max_entity_types, logger=logger, epochs=config.model.epochs, weight_decay=config.model.weight_decay, learning_rate=config.model.learning_rate, saved_model=config.model.saved_model, dropout=config.model.dropout, device=device)


    def __init_tracker(self):
        '''tracker'''
        self.tracker =  Dialog_Tracker(self.bert_model_name, len(self.reader), self.max_entity_types, self.fc_response1, self.fc_response2, self.dropout)
        
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
        self.tracker.train()
        
        for epoch, d in enumerate(self.reader):
            self.tracker.zero_grad()
            y_prob = torch.log(self.tracker(d))
        
            responses = d['response']

            loss = self.loss_function(y_prob, responses)
        
            #precision
            _, y_pred = torch.max(y_prob.data, 1)
            
            precision = y_pred.eq(responses).sum().item()/responses.numel()
            
            if self.logger: self.logger.info('{} {} {} {}'.format(epoch, self.epochs, loss.item(), precision))
        
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


