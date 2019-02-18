#!/usr/bin/env python
import torch.nn as nn
from .generative_tracker import Generative_Tracker
from ..skills.generative_response import Generative_Response


'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Generative_Supervised(nn.Module):
    '''
        Generative based chatbot
    '''
    def __init__(self, reader):
        self.reader = reader
   

    @classmethod
    def build(cls, config, reader_cls):
        '''
            construct model from config

            Input:
                - config: configure dictionary
                - reader_cls: class for reader
        '''
        logger = setLogger(**config.logger)
        ner = NER(**config.ner)
        tokenizer = Tokenizer_BERT(**config.tokenizer) 
        vocab = tokenizer.vocab

        #skill
        response = Generative_Response(tokenizer=tokenizer) 
        topic_manager = Topic_Manager()
        topic_manager.register(config.model.skill_name, response)

          

