#!/usr/bin/env python
import sys, numpy
from .skill_base import Skill_Base
from ..model.generative_tracker import Generative_Tracker

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Generative_Response(Skill_Base):
    '''
        Generative skill for chatbot
        
        Input:
            - tokenizer: instance of nlptools.text.tokenizer
            - vocab:  instance of nlptools.text.vocab
            - max_seq_len: int, maximum sequence length
    '''

    def __init__(self, tokenizer, vocab,  max_seq_len=100, **args):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len


    def __getitem__(self, response):
        '''
            Predeal response string
        '''
        response_tokens = self.tokenizer(response)
        response_ids = self.vocab.words2id(response_tokens)
        return response_ids


    def init_model(self, saved_model="generative_tracker.pt", device='cpu', **args):
        '''
            Initialize model
            
            Input:
                - saved_model: str, default is "dialog_tracker.pt"
                - device: string, model location, default is 'cpu'
                - see ..model.generative_tracker.Generative_Tracker for more parameters if path of saved_model not existed
        '''
        if os.path.exists(saved_model):
            self.checkpoint = torch.load(self.saved_model, map_location=lambda storage, location: torch.device(device))
            self.model = Generative_Tracker(**self.checkpoint['config_model']) 
            self.model.to(device)
            self.model.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.model = Generative_Tracker(**args)
            self.model.to(device)

    
    def get_response(self, current_status):
        '''
            predict response value from current status

            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        '''
        if self.model.training:
            return self.model(current_status)
        else:
            pass


    def update_response(self, skill_name, response, current_status):
        '''
            update current response to the response status.
            
            Input:
                - skill_name: string, name of current skill
                - response: value of response
                - current_status: dictionary of status, generated from Dialog_Status module
        '''
        
        current_status['response_string'] = self.vocab.id2words(response) 
        response = response[:self.max_seq_len-2]
        seq_len = len(response) + 2
        response_key = 'response_' + skill_name
        mask_key = 'response_mask_' + skill_name
        
        current_status[response_key] = numpy.zeros(self.max_seq_len, 'int')
        current_status[mask_key] = numpy.zeros(self.max_seq_len, 'int')
        
        current_status[response_key][0] = self.vocab.CLS_ID
        current_status[response_key][1:seq_len-1] = response
        current_status[response_key][seq_len-1] = self.vocab.SEP_ID
        
        current_status[mask_key][:seq_len] = 1

        return current_status

