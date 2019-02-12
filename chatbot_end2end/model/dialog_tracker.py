#!/usr/bin/env python
import time
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.utils.rnn import PackedSequence
from .sentence_encoder import Sentence_Encoder
from pytorch_pretrained_bert.modeling import BertModel

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Dialog_Tracker(nn.Module):
    '''
        dialog tracker for end2end chatbot 

        Input:
            - skill_name: string, current skill name
            - bert_model_name: bert model file location or one of the supported model name
            - Nresponses: number of available responses
            - kernel_num: int
            - kernel_size: int
            - max_entity_types: int
            - fc_responses: int for int list, default is 5
            - entity_layers: int, default is 2
            - lstm_layers: int, default is 1
            - dropout: float, default is 0.2
            
    '''
    def __init__(self, skill_name, bert_model_name, Nresponses, max_entity_types, entity_layers=2, entity_emb_dim=50, lstm_layers=1, hidden_dim=300, dropout=0.2):
        super().__init__()
        self.skill_name = skill_name
        self.encoder = Sentence_Encoder(bert_model_name)

        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(2)

        fc_entity_layers = [nn.Linear(max_entity_types, max_entity_types) for i in range(entity_layers-1)]
        fc_entity_layers.append(nn.Linear(max_entity_types, entity_emb_dim))
        self.fc_entity = nn.Sequential(*fc_entity_layers)

        self.fc_dialog = nn.Linear(self.encoder.hidden_size + entity_emb_dim + 1, hidden_dim)
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, Nresponses)
        self.loss_function = torch.nn.NLLLoss()
        self.softmax = nn.Softmax(dim=1)

    def entityencoder(self, x):
        '''
            entity encoder, model framwork:
                - linear + linear 

            Input:
                - onehot present of entity names
        '''
        x = self.fc_entity(x)
        x = self.dropout(x)
        return x

    
    def dialog_embedding(self, utterance, utterance_mask, entity,  sentiment):
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
        sequence_output, pooled_output = self.encoder(utterance, attention_mask=utterance_mask, output_all_encoded_layers=False)
        
        #entity name embedding
        entity = self.entityencoder(entity) 
        
        sentiment = sentiment.unsqueeze(1)
    
        #concat together and apply linear
        utter = torch.cat((pooled_output, entity, sentiment), 1)
        
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
        pack_batch = dialogs['utterance'].batch_sizes

        dialog_emb = self.dialog_embedding(dialogs['utterance'].data, dialogs["utterance_mask"].data, dialogs['entity'].data, dialogs["sentiment"].data)
        
        dialog_emb = PackedSequence(dialog_emb, pack_batch) #feed batch_size and pack to packedsequence
        
        #dialog embedding to lstm as dialog tracker

        lstm_out, (ht, ct) = self.lstm(dialog_emb)
        
        lstm_out = self.dropout(lstm_out.data)
        
        hidden = self.fc_out(lstm_out)
        #output to softmax
        lstm_softmax = self.softmax(hidden)
        
        #apply mask 
        response = lstm_softmax * dialogs['response_mask'][self.skill_name].data + 1e-15
        
        y_prob = torch.log(response)

        if 'response' in dialogs:
            loss = self.loss_function(y_prob, dialogs['response'][self.skill_name].data)
            return y_prob, loss

        return y_prob


