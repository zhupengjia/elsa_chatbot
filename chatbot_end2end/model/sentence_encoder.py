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
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_model_name)
        self.embedding = self.encoder.embedding
        self.config = self.encoder.config
        for param in self.encoder.parameters():
            param.requires_grad = False # freeze bert parameter
        self.hidden_size = self.config.hidden_size

    def forward(self, sentence, attention_mask, output_all_encoded_layers=False):
        sequence_output, pooled_output = self.encoder(sentence, attention_mask=attention_mask, output_all_encoded_layers=False)
        return sequence_output, pooled_output

