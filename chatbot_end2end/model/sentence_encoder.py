#!/usr/bin/env python
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
from nlptools.zoo.encoders.transformer import TransformerEncoder

class Sentence_Encoder(nn.Module):
    '''
        Sentence encoder, use BERT

        Input:
            - bert_model_name: bert model file location or one of the supported model name
    '''
    def __init__(self, bert_model_name=None, vocab_size=30522, encoder_hidden_layers=12,
                 encoder_attention_heads=12, max_position_embeddings=512,
                 encoder_intermediate_size=3072, encoder_hidden_size=768, dropout=0.1, **args):
        super().__init__()
        if bert_model_name is not None:
            self.encoder = TransformerEncoder.from_pretrained(bert_model_name)
            self.bert_model_name = bert_model_name
        else:
            self.encoder = TransformerEncoder(vocab_size=vocab_size,
                                              num_hidden_layers=encoder_hidden_layers,
                                              num_attention_heads=encoder_attention_heads,
                                              max_position_embeddings=max_position_embeddings,
                                              intermediate_size=encoder_intermediate_size,
                                              hidden_size=encoder_hidden_size,
                                              dropout=0.1)
        self.config = {"bert_model_name": bert_model_name,
                       "vocab_size": self.encoder.config.vocab_size,
                       "encoder_hidden_layers": self.encoder.config.num_hidden_layers,
                       "encoder_attention_heads": self.encoder.config.num_attention_heads,
                       "max_position_embeddings": self.encoder.config.max_position_embeddings,
                       "encoder_intermediate_size": self.encoder.config.intermediate_size,
                       "encoder_hidden_size": self.encoder.config.hidden_size}
        self.embedding = self.encoder.embeddings
        self.hidden_size = self.config["encoder_hidden_size"]

    def freeze(self):
        """
            freeze encoder parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = False # freeze bert parameter

    def forward(self, sentence, attention_mask, output_all_encoded_layers=False):
        sequence_output, pooled_output = self.encoder(sentence, attention_mask=attention_mask, output_all_encoded_layers=False)
        return sequence_output, pooled_output

