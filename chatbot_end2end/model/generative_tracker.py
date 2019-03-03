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
    def __init__(self, skill_name, encoder, decoder_hidden_layers=1, decoder_attention_heads=2, decoder_hidden_size=1024, dropout=0.2):
        super().__init__()
        self.response_key = 'response_' + skill_name
        self.mask_key = 'response_mask_' + skill_name
        self.encoder = encoder
        self.decoder = TransformerDecoder(self.encoder.embedding, decoder_hidden_layers, decoder_attention_heads, decoder_hidden_size, dropout) 
        self.loss_function = nn.NLLLoss(ignore_index=0)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, dialogs, train_mode=True):
        pack_batch = dialogs['utterance'].batch_sizes
        
        #utterance embedding
        sequence_output, pooled_output = self.encoder(dialogs['utterance'].data, attention_mask=dialogs['utterance_mask'].data, output_all_encoded_layers=False)
       
        prev_output = dialogs[self.response_key].data[:, :-1]
        target_output = dialogs[self.response_key].data[:, 1:]
        
        print("="*60)

        output, attn = self.decoder(prev_output, sequence_output, dialogs['utterance_mask'].data)
        print(output.size())
        output_probs = self.logsoftmax(output)
       
        target_masks = dialogs[self.mask_key].data[:, 1:]
        target_masks = target_masks.unsqueeze(-1).expand_as(output_probs)


        if train_mode:
            target_masks = target_masks.unsqueeze(-1).contiguous().view(-1, target_masks.size(2))
            target_output = target_output.unsqueeze(-1).contiguous().view(-1)
            
            output_probs_expand = output_probs.contiguous().view(-1, output_probs.size(2))
           
            loss = self.loss_function(output_probs_expand, target_output)
            return output_probs, loss
        
        return output_probs
        

