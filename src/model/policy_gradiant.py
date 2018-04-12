#!/usr/bin/env python
import sys, torch, os, numpy
from .rule_based import Rule_Based
from .dialog_tracker import Dialog_Tracker
from ..module.dialog_status import Dialog_Status
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from ailab.utils import Config, setLogger
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


    def __init_reader(self, reader_cls):
        #reader
        self.reader = reader_cls(self.cfg)
        self.reader.build_responses() #build response template index, will read response template and create entity need for each response template every time, but not search index.
        self.reader.responses.build_mask()


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
            #self.logger.info('--------- new dialog ------')
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
                #self.logger.info('{}\t{}'.format(utterance, response))

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


            if epoch > 0 and epoch%1000 == 0: 
                model_state_dict = self.tracker.state_dict()
                optimizer_state_dict = self.optimizer.state_dict()
                checkpoint = {
                    'model': model_state_dict,
                    'optimizer': optimizer_state_dict,
                }
                torch.save(checkpoint, self.cfg.model['saved_model'])
            
             
