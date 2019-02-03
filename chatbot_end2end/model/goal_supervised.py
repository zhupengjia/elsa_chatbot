#!/usr/bin/env python
import sys, torch, os, numpy
from .dialog_tracker import Dialog_Tracker
from nlptools.utils import Config, setLogger
from ..module.entity_dict import Entity_Dict
from nlptools.text.tokenizer import Tokenizer_BERT
from nlptools.text.ner import NER
import torch.optim as optim
from ..module.topic_manager import Topic_Manager
from ..skills.goal_response import Goal_Response
from nltk.sentiment.vader import SentimentIntensityAnalyzer

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Goal_Supervised:
    def __init__(self, reader, bert_model_name, max_entity_types, fc_responses=5, entity_layers=2, lstm_layers=1, hidden_dim=300, epochs=1000, weight_decay=0, learning_rate=0.001, saved_model="model.pt", dropout=0.2, device=None, logger=None):
        '''
            Supervised learning for end2end goal oriented chatbot

            Input:
                - bert_model_name: bert model file location or one of the supported model name
                - reader: reader instance, see ..module.reader.*.py
                - max_entity_types: int, number of entity types 
                - fc_responses: int
                - entity_layers: int, default is 2
                - lstm_layers: int, default is 1
                - hidden_dim: int, default is 300
                - epochs: int, default is 1000
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
        self.fc_responses = fc_responses
        self.lstm_layers = lstm_layers
        self.entity_layers = entity_layers
        self.hidden_dim = hidden_dim
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


        #skill
        goal_response = Goal_Response(tokenizer=tokenizer, hook=hook, template_file=config.response_template)
        topic_manager = Topic_Manager()
        topic_manager.register(config.skill_name, goal_response)
        sentiment_analyzer = SentimentIntensityAnalyzer()



        entity_dict = Entity_Dict(**config.entity_dict) 
       
        #reader
        reader = reader_cls(tokenizer=tokenizer, ner=ner, entity_dict=entity_dict, hook=hook, max_entity_types=config.entity_dict.max_entity_types, max_seq_len=config.model.max_seq_len, epochs=config.model.epochs, batch_size=config.model.batch_size, device=device, logger=logger)



        return cls(reader=reader, logger=logger, bert_model_name=config.tokenizer.bert_model_name, max_entity_types=config.entity_dict.max_entity_types, fc_responses=config.model.fc_responses, entity_layers=config.model.entity_layers, lstm_layers=config.model.lstm_layers, hidden_dim=config.model.hidden_dim, epochs=config.model.epochs, weight_decay=config.model.weight_decay, learning_rate=config.model.learning_rate, saved_model=config.model.saved_model, dropout=config.model.dropout, device=device)


    def __init_tracker(self):
        '''tracker'''
        self.topic_manager = Topic_Manager()
        
        self.tracker =  Dialog_Tracker(bert_model_name=self.bert_model_name, Nresponses=len(self.reader), max_entity_types=self.max_entity_types, fc_responses=self.fc_responses, entity_layers=self.entity_layers, lstm_layers=self.lstm_layers, hidden_dim=self.hidden_dim, dropout=self.dropout)
        
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
            
            self.logger.info('{} {} {} {}'.format(epoch, self.epochs, loss.item(), precision))
        
            loss.backward()
            self.optimizer.step()

            #save
            if epoch > 0 and epoch%1000 == 0: 
                model_state_dict = self.tracker.state_dict()
                torch.save(model_state_dict, self.saved_model)


