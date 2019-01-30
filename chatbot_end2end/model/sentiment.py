#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Sentiment(nn.Module):
    '''
        Sentiment

        Input:
            - encoder_hidden_size: hidden size of encoder 
            - nlayers: layers of linear, default is 5
            
    '''
    def __init__(self, encoder_hidden_size, nlayers=5):
        super().__init__()
        self.hidden_size = encoder_hidden_size
        fc_layers = []
        self.num_labels = 5 # from amazon review data
        for i in range(nlayers-1):
            fc_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        fc_layers.append(nn.Linear(self.hidden_size, self.num_labels))
        self.classifier = nn.Sequential(*fc_layers) 

    def forward(self, sentence_pooled_output, labels=None):
        '''
            Input:
                - sentence_pooled_output: pooled_output from bert encoder
                - labels: target labels, default is None
        '''
        logits = self.classifier(sentence_pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

