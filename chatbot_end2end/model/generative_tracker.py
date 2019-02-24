#!/usr/bin/env python
import torch
import torch.nn as nn


'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Generative_Tracker(nn.Module):
    '''
        Generative based chatbot

        Input:
            - skill_name: string, current skill name
            - encoder: sentence encoder instance from .sentence_encoder
    '''
    def __init__(self, skill_name, encoder, dropout=0.2):
        super().__init__()
        self.response_key = 'response_' + skill_name
        self.mask_key = 'response_mask_' + skill_name
        self.encoder = encoder
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, dialogs):
        pack_batch = dialogs['utterance'].batch_sizes
        
        #utterance embedding
        sequence_output, pooled_output = self.encoder(dialogs['utterance'].data, attention_mask=dialogs['utterance_mask'].data, output_all_encoded_layers=False)


        print(pooled_output.size())
        return 0
        

