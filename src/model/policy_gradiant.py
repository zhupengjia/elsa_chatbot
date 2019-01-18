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
    def __init__(self, cfg_fn, reader_cls, hook_cls):
        '''
            Policy Gradiant for end2end chatbot

            Input:
                - cfg_fn: config file location
                - reader_class: class for reader
                - hook_cls: class for hook
        '''
        self.cfg = Config(cfg_fn)
        if not torch.cuda.is_available(): 
            self.cfg.model.use_gpu = 0
        self.logger = setLogger(self.cfg.logger)
        
        self.__init_adversary(hook_cls)
        self.__init_reader(reader_cls)
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

        return cls(reader=reader, logger=logger, bert_model_name=config.tokenizer.bert_model_name, max_entity_types=config.entity_dict.max_entity_types, fc_responses=config.model.fc_responses, entity_layers=config.model.entity_layers, lstm_layers=config.model.lstm_layers, hidden_dim=config.model.hidden_dim, epochs=config.model.epochs, weight_decay=config.model.weight_decay, learning_rate=config.model.learning_rate, saved_model=config.model.saved_model, dropout=config.model.dropout, device=device)

    

    def __init_adversary(self, hook_cls):
        '''adversary chatbot'''
        hook = hook_cls(self.cfg)
        self.ad_chatbot = Rule_Based(self.cfg.rule_based, hook)
        self.clientid = 'DLTRAIN' 


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
        
        self.optimizer = optim.Adam(self.tracker.parameters(), lr=self.cfg.model.learning_rate, weight_decay=self.cfg.model.weight_decay)


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
            
             
