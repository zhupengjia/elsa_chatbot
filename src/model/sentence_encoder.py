#!/usr/bin/env python
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel

class Sentence_Encoder(nn.Module):
    '''
        Sentence encoder, use BERT

        Input:
            - bert_model_name: bert model file location or one of the supported model name
    '''
    def __init__(self, bert_model_name):
        self.encoder = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, sentence, attention_mask, output_all_encoded_layers=False):
        sequence_output, pooled_output = self.encoder(utterance, attention_mask=attention_mask, output_all_encoded_layers=False)
        return sequence_output, pooled_output

