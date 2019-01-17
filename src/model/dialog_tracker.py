#!/usr/bin/env python
import time
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.utils.rnn import PackedSequence
from .model_base import Model_Base
from pytorch_pretrained_bert.modeling import BertModel

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Dialog_Tracker(Model_Base):
    '''
        dialog tracker for end2end chatbot 

        Input:
            - bert_model_name: bert model file location or one of the supported model name
            - Nresponses: number of available responses
            - kernel_num: int
            - kernel_size: int
            - max_entity_types: int
            - fc_response1: int
            - fc_response2: int
            - dropout: float
            
    '''
    def __init__(self, bert_model_name, Nresponses, max_entity_types, fc_response1, fc_response2, dropout=0.2):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_model_name) 
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(2)
        self.fc_entity1 = nn.Linear(max_entity_types,max_entity_types)
        self.fc_entity2 = nn.Linear(max_entity_types,max_entity_types)
        self.fc_response1 = nn.Linear(Nresponses, fc_response1)
        self.fc_response2 = nn.Linear(fc_response1, fc_response2)
        self.fc_dialog = nn.Linear(fc_response1, Nresponses)
        self.lstm = nn.LSTM(Nresponses, Nresponses, num_layers=1, batch_first=True)
        self.softmax = nn.Softmax(dim=1)

    def entityencoder(self, x):
        '''
            entity encoder, model framwork:
                - linear + linear 

            Input:
                - onehot present of entity names
        '''
        x = self.fc_entity1(x)
        x = self.fc_entity2(x)
        x = self.dropout(x)
        return x

    def responseencoder(self, x):
        '''
            response encoder, model framework:
                - linear + linear 

            Input:
                - onehot present of response
        '''
        x = self.fc_response1(x)
        x = self.fc_response2(x)
        x = self.dropout(x)
        return x


    def dialog_embedding(self, utterance, entity,  response_prev):
        '''
            Model framework:
                - utterance_embedding + entityname_embedding + prev_response embedding -> linear

            Get dialog embedding from utterance, entity, response_prev

            Input:
                - utterance, entity, response_prev are from three related keys of dialog_status.torch output

            Output:
                - dialog embedding
        '''
        #utterance embedding
        utterance = self.encoder(utterance) 
        utter_att = self.attention(utterance, utterance) 
        #entity name embedding
        entity = self.entityencoder(entity) 
        #previous response embedding
        response_prev = self.responseencoder(response_prev)
        #concat together and apply linear
        utter = torch.cat((utter_att, entity, response_prev), 1)
        emb = self.fc_dialog(utter)
        return emb


    def forward(self, dialogs):
        '''
            Model framework:
                - dialogs -> dialog_embedding -> lstm -> softmax*mask -> logsoftmax
            
            Input:
                - dialogs: output from dialog_status.torch
            
            output:
                - logsoftmax
        '''
        #first get dialog embedding
        dialog_emb = self.dialog_embedding(dialogs['utterance'], dialogs['entity'], dialogs['response_prev'])
        dialog_emb = PackedSequence(dialog_emb, dialogs['batch_sizes']) #feed batch_size and pack to packedsequence
        #dialog embedding to lstm as dialog tracker
        lstm_out, (ht, ct) = self.lstm(dialog_emb)
        #output to softmax
        lstm_softmax = self.softmax(lstm_out.data)
        #apply mask 
        response = lstm_softmax * dialogs['mask'] + 1e-15
        return response


