#!/usr/bin/env python
import time
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.utils.rnn import PackedSequence
from .sentence_encoder import Sentence_Encoder
from .model_base import Model_Base

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Dialog_Tracker(Model_Base):
    '''
        dialog tracker for end2end chatbot 

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - cnn_kernel_num
                    - cnn_kernel_size
                    - dropout
                    - max_entity_types
                    - fc_response1
                    - fc_response2
            - vocab: instance of ailab.text.vocab
            - Nresponses: number of available responses
            
    '''
    def __init__(self, cfg, vocab, Nresponses):
        super().__init__(cfg, vocab)
        self.Nresponses = Nresponses
        self.network()

    def network(self):
        '''
            Define the network modules
        '''
        self.encoder = Sentence_Encoder(self.cfg, self.vocab) # sentence encoder for utterance embedding
        self.encoder.network()
        self.conv = nn.Conv2d(in_channels = 1, \
                out_channels = self.cfg['cnn_kernel_num'], \
                kernel_size = (self.cfg['cnn_kernel_size'], self.vocab.emb_ins.vec_len),\
                padding = 0)
        self.dropout = nn.Dropout(self.cfg['dropout'])
        self.pool = nn.AvgPool1d(2)
        self.fc_entity1 = nn.Linear(self.cfg['max_entity_types'],self.cfg['max_entity_types'])
        self.fc_entity2 = nn.Linear(self.cfg['max_entity_types'],self.cfg['max_entity_types'])
        self.fc_response1 = nn.Linear(self.Nresponses, self.cfg['fc_response1'])
        self.fc_response2 = nn.Linear(self.cfg['fc_response1'], self.cfg['fc_response2'])
        fc1_input_size = self.cfg['cnn_kernel_num']*2 + self.cfg['max_entity_types'] + self.cfg['fc_response2']
        self.fc_dialog = nn.Linear(fc1_input_size, self.Nresponses)
        self.lstm = nn.LSTM(self.Nresponses, self.Nresponses, num_layers=1, dropout = self.cfg['dropout'], batch_first=True)
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


