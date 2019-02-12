#!/usr/bin/env python
import sys, torch, os, numpy
from .dialog_tracker import Dialog_Tracker
from nlptools.utils import Config, setLogger
from nlptools.text.tokenizer import Tokenizer_BERT
from nlptools.text.ner import NER
import torch.optim as optim
from ..module.topic_manager import Topic_Manager
from ..skills.goal_response import Goal_Response
from ..module.nltk_sentiment import NLTK_Sentiment
from ..module.dialog_status import Collate_Fn
from torch.utils.data import DataLoader

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Goal_Supervised:
    def __init__(self, reader, bert_model_name, max_entity_types=1024, Nresponses=10, fc_responses=5, entity_layers=2, lstm_layers=1, hidden_dim=300, batch_size=100, num_workers=1, epochs=1000, weight_decay=0, learning_rate=0.001, saved_model="model.pt", dropout=0.2, device='cpu', logger=None):
        '''
            Supervised learning for end2end goal oriented chatbot

            Input:
                - reader: reader instance, see ..module.reader.*.py
                - bert_model_name: bert model file location or one of the supported model name
                - fc_responses: int
                - entity_layers: int, default is 2
                - lstm_layers: int, default is 1
                - hidden_dim: int, default is 300
                - epochs: int, default is 1000
                - weight_decay: int, default is 0
                - learning_rate: float, default is 0.001
                - saved_model: str, default is "model.pt"
                - dropout: float, default is 0.2
                - device: string of torch device, default is "cpu"
                - logger: logger instance ,default is None
        '''
        self.bert_model_name = bert_model_name
        self.saved_model = saved_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_entity_types = max_entity_types
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.Nresponses = Nresponses
        self.device = torch.device(device)
        self.logger = logger 

        self.reader = reader
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
        vocab = tokenizer.vocab
        
        #skill
        goal_response = Goal_Response(tokenizer=tokenizer, hook=hook, template_file=config.response_template)
        topic_manager = Topic_Manager()
        topic_manager.register(config.skill_name, goal_response)

        #reader
        reader = reader_cls(vocab=vocab, tokenizer=tokenizer, ner=ner, topic_manager=topic_manager, sentiment_analyzer=NLTK_Sentiment(), max_seq_len=config.reader.max_seq_len, max_entity_types=config.model.max_entity_types, logger=logger)
        reader.read(config.reader.train_data)

        return cls(reader=reader, logger=logger, bert_model_name=config.tokenizer.bert_model_name, Nresponses=len(goal_response), **config.model)


    def __init_tracker(self):
        '''tracker'''
        self.tracker =  Dialog_Tracker(bert_model_name=self.bert_model_name, Nresponses=self.Nresponses, max_entity_types=self.max_entity_types, fc_responses=self.fc_responses, entity_layers=self.entity_layers, lstm_layers=self.lstm_layers, hidden_dim=self.hidden_dim, dropout=self.dropout)
        
        self.tracker.to(self.device)

        #checkpoint
        if os.path.exists(self.saved_model):
            checkpoint = torch.load(self.saved_model, map_location=lambda storage, location: self.device)
            self.tracker.load_state_dict(checkpoint['model'])
        

        self.loss_function = torch.nn.NLLLoss()
        self.optimizer = optim.Adam(self.tracker.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.generator = DataLoader(self.reader, batch_size=self.batch_size, collate_fn=Collate_Fn, shuffle=True, num_workers=self.num_workers)



    def read(self, filepath):
        '''
            read train data

            Input:
                - filepath: file location
        '''
        self.reader.read(filepath)


    def train(self):
        self.tracker.train() #set train flag
        
        for epoch in range(self.epochs):
            for it, d in enumerate(self.generator):

                d.to(self.device)
                print(d)
                sys.exit()
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


