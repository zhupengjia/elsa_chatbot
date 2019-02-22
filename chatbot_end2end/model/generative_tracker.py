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
        self.skill_name = skill_name
        self.encoder = encoder
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, dialogs):
        return 0


