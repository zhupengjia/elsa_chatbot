#!/usr/bin/env python
import os, numpy, torch
from .skill_base import SkillBase
from ..model.generative_tracker import Generative_Tracker
from ..module.dialog_status import format_sentence

"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""


class GenerativeResponse(SkillBase):
    """
        Generative skill for chatbot
        
        Input:
            - tokenizer: instance of nlptools.text.tokenizer
            - vocab:  instance of nlptools.text.vocab
            - max_seq_len: int, maximum sequence length
    """

    def __init__(self, skill_name, tokenizer, vocab,  max_seq_len=100, beam_size=1, **args):
        super(GenerativeResponse, self).__init__(skill_name)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.beam_size = beam_size

    def __getitem__(self, response):
        """
            Predeal response string
        """
        response_tokens = self.tokenizer(response)
        response_ids = self.vocab.words2id(response_tokens)
        return response_ids

    def init_model(self, saved_model="generative_tracker.pt", device='cpu', **args):
        """
            Initialize model
            
            Input:
                - saved_model: str, default is "dialog_tracker.pt"
                - device: string, model location, default is 'cpu'
                - see ..model.generative_tracker.Generative_Tracker for more parameters if path of saved_model not existed
        """
        additional_args = {"beam_size": self.beam_size}
        args = {**args, **additional_args}
        if os.path.exists(saved_model):
            self.checkpoint = torch.load(saved_model, map_location=lambda storage, location: storage)
            self.model = Generative_Tracker(**{**args, **self.checkpoint['config_model']}) 
            self.model.to(device)
            self.model.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.model = Generative_Tracker(**args)
            self.model.to(device)

    def eval(self):
        """
        Set model to eval mode
        """
        self.model.eval()

    def get_response(self, status_data):
        """
            predict response value from status

            Input:
                - status_data: data converted from dialog status
        """
        if self.model.training:
            return self.model(status_data)
        else:
            return self.model(status_data)

    def update_response(self, response, current_status):
        """
            update current response to the response status.
            
            Input:
                - response: value of response
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        
        current_status["entity"]['RESPONSE'] = self.vocab.id2words(response) 
        response_key = 'response_' + self.skill_name
        mask_key = 'response_mask_' + self.skill_name
        
        current_status[response_key], self.current_status[mask_key] =\
                format_sentence(response, vocab=vocab, max_seq_len=self.max_seq_len)
        
        return current_status

