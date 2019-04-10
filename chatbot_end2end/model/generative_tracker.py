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
    def __init__(self, skill_name, encoder, decoder_hidden_layers=1, decoder_attention_heads=2, decoder_hidden_size=1024, dropout=0, **args):
        super().__init__()
        self.config = {
                    "bert_model_name": encoder.bert_model_name,
                    "decoder_hidden_layers": decoder_hidden_layers,
                    "decoder_attention_heads": decoder_attention_heads,
                    "decoder_hidden_size": decoder_hidden_size
                }

        self.response_key = 'response_' + skill_name
        self.mask_key = 'response_mask_' + skill_name
        self.encoder = encoder
        embedding_dim = self.encoder.embedding.word_embeddings.embedding_dim 
        self.control_layer = nn.Linear(embedding_dim+1, embedding_dim)
        self.decoder = TransformerDecoder(self.encoder.embedding, decoder_hidden_layers, decoder_attention_heads, decoder_hidden_size, dropout) 
        self.loss_function = nn.NLLLoss(ignore_index=0) #ignore padding loss
        self.logsoftmax = nn.LogSoftmax(dim=2)


    def dialog_embedding(self, utterance, utterance_mask, sentiment):
        #utterance embedding
        sequence_output, pooled_output = self.encoder(utterance, attention_mask=utterance_mask, output_all_encoded_layers=False)

        #sentiment
        sentiment = sentiment.unsqueeze(1).expand(-1, sequence_output.size(1), 1)

        #combine together
        sequence_out = torch.cat((sequence_output, sentiment), 2)
        sequence_out = self.control_layer(sequence_out)

        return sequence_out


    def forward(self, dialogs):
        #encoder
        encoder_out = self.dialog_embedding(dialogs['utterance'].data, dialogs["utterance_mask"].data, dialogs["sentiment"].data)

        prev_output = dialogs[self.response_key].data[:, :-1]

        print(encoder_out)
        print(prev_output)
        sys.exit()

        #decoder
        output, attn = self.decoder(prev_output, encoder_out, dialogs['utterance_mask'].data)
        output_probs = self.logsoftmax(output)

        #target_masks = dialogs[self.mask_key].data[:, 1:]
        #target_masks = target_masks.unsqueeze(-1).expand_as(output_probs)

        if self.training:
            #target_masks = target_masks.unsqueeze(-1).contiguous().view(-1, target_masks.size(2))
            target_output = dialogs[self.response_key].data[:, 1:]
            target_output = target_output.unsqueeze(-1).contiguous().view(-1)

            output_probs_expand = output_probs.contiguous().view(-1, output_probs.size(2))

            loss = self.loss_function(output_probs_expand, target_output)
            return output_probs, loss

        return output_probs

