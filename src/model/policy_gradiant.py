#!/usr/bin/env python
import sys, torch, os, numpy
from .rule_based import Rule_Based
from .dialog_tracker import Dialog_Tracker
from ..module.dialog_status import Dialog_Status
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from nlptools.utils import Config, setLogger
import torch.optim as optim

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Policy_Gradiant:
    def __init__(self, adversal_chatbot, reader, bert_model_name, max_entity_types, fc_responses=5, entity_layers=2, lstm_layers=1, hidden_dim=300, epochs=1000, weight_decay=0, learning_rate=0.001, saved_model="model.pt", dropout=0.2, device=None, logger=None):
        '''
            Policy Gradiant for end2end chatbot

            Input:
                - adversal_chatbot: adversal chatbot instance
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
        self.cfg = Config(cfg_fn)
        if not torch.cuda.is_available(): 
            self.cfg.model.use_gpu = 0
       
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

        self.ad_chatbot = adversal_chatbot
        self.clientid = 'DLTRAIN'

        self.__init_tracker()

    @classmethod
    def build(cls, config, reader_cls, hook, adversal_hook):
        '''
            construct model from config

            Input:
                - config: configure dictionary
                - reader_cls: class for reader
                - hook: hook instance, please check src/hook/babi_gensays.py for example
                - adversal_hook: adversal hook instance, please check src/hook/babi_gensays.py for example

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

        #adversal chatbot
        ad_chatbot = Rule_Based.build(config.rule_based, adversal_hook)

        return cls(adversal_chatbot = ad_chatbot, reader=reader, logger=logger, bert_model_name=config.tokenizer.bert_model_name, max_entity_types=config.entity_dict.max_entity_types, fc_responses=config.model.fc_responses, entity_layers=config.model.entity_layers, lstm_layers=config.model.lstm_layers, hidden_dim=config.model.hidden_dim, epochs=config.model.epochs, weight_decay=config.model.weight_decay, learning_rate=config.model.learning_rate, saved_model=config.model.saved_model, dropout=config.model.dropout, device=device)


    def __init_tracker(self):
        '''tracker'''
        self.tracker =  Dialog_Tracker(bert_model_name=self.bert_model_name, Nresponses=len(self.reader), max_entity_types=self.max_entity_types, fc_responses=self.fc_responses, entity_layers=self.entity_layers, lstm_layers=self.lstm_layers, hidden_dim=self.hidden_dim, dropout=self.dropout)
        
        self.tracker.to(self.device)

        #checkpoint
        if os.path.exists(self.saved_model):
            checkpoint = torch.load(self.saved_model, map_location={'cuda:0': self.device.type})
            self.tracker.load_state_dict(checkpoint['model'])
        

        self.loss_function = torch.nn.NLLLoss()
        self.optimizer = optim.Adam(self.tracker.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


    def __memory_replay(self):
        '''memory replay'''
        dialogs = []
        N_true = 0
        for batch in range(self.cfg.model.batch_size):
            self.logger.debug('--------- new dialog ------')
            #reset adversal chatbot
            self.ad_chatbot.reset(self.clientid)
            #greeting utterance
            utterance = self.ad_chatbot.get_reply('', self.clientid)
            #new dialog status
            dialog_status = self.reader.new_dialog()
            #dialog loop
            for loop in range(self.cfg.model.rl_maxloop):
                dialog_status.add_utterance(utterance)
                dialog_status.getmask()
                data = Dialog_Status.torch(self.cfg, self.reader.vocab, self.reader.entity_dict, [dialog_status])
                
                y_prob = self.tracker(data)

                _, y_pred = torch.max(y_prob.data, 1)
                if self.cfg.model.use_gpu:
                    y_pred = y_pred.cpu()
                y_pred = int(y_pred.numpy()[-1])
                response = self.reader.get_response(y_pred)
                self.logger.debug('{}\t{}'.format(utterance, response))

                dialog_status.add_response(y_pred)
               
                if y_pred == len(self.reader) - 1:
                    N_true += 1
                    break
            
                #get new utterance
                utterance = self.ad_chatbot.get_reply(response, self.clientid)
            dialogs.append(dialog_status) 
            
        
        #convert to torch variable
        dialogs = Dialog_Status.torch(self.cfg, self.reader.vocab, self.reader.entity_dict, dialogs)

        return dialogs, float(N_true)/self.cfg.model.batch_size


    def train(self):
        '''
            Train the model. No input needed
        '''
        for epoch in range(self.cfg.model.epochs):
            dialogs, precision = self.__memory_replay()
            self.tracker.zero_grad()
            y_prob = self.tracker(dialogs)
            
            self.logger.info('{} {} {}'.format(epoch, self.cfg.model.epochs, precision))
            
            m = torch.distributions.Categorical(y_prob)
            action = m.sample()
            
            loss = -m.log_prob(action) * dialogs['reward']
    
            loss.sum().backward()
            self.optimizer.step()


            if epoch > 0 and epoch%10 == 0: 
                model_state_dict = self.tracker.state_dict()
                optimizer_state_dict = self.optimizer.state_dict()
                checkpoint = {
                    'model': model_state_dict,
                    'optimizer': optimizer_state_dict,
                }
                torch.save(checkpoint, self.cfg.model['saved_model'])
            
             
