#!/usr/bin/env python
import torch
import torch.nn as nn
from nlptools.zoo.encoders.transformer_decoder import TransformerDecoder

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
    def __init__(self, skill_name, encoder, decoder_hidden_layers=6, decoder_attention_heads=8, decode_intermediate_size=1024, dropout=0.2):
        super().__init__()
        self.response_key = 'response_' + skill_name
        self.mask_key = 'response_mask_' + skill_name
        self.encoder = encoder
        self.decoder = TransformerDecoder(self.encoder.embedding, decoder_hidden_layers, decoder_attention_heads, decode_intermediate_size, dropout) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, dialogs):
        pack_batch = dialogs['utterance'].batch_sizes
        
        #utterance embedding
        sequence_output, pooled_output = self.encoder(dialogs['utterance'].data, attention_mask=dialogs['utterance_mask'].data, output_all_encoded_layers=False)


        print(pooled_output.size())
        return 0
        

