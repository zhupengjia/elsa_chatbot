#!/usr/bin/env python
import torch
import torch.nn as nn

class SentenceEncoder(nn.Module):
    '''
        Sentence encoder, use BERT

        Input:
            - bert_model_name: bert model file location or one of the supported model name
    '''
    def __init__(self, model_type="transformer", bert_model_name=None, vocab_size=30522,
                 pretrained_embedding=None, encoder_hidden_layers=12, encoder_hidden_size=768,
                 encoder_intermediate_size=1024, encoder_attention_heads=12,
                 max_position_embeddings=512,  dropout=0.1, **args):
        super().__init__()
        if model_type == "gru":
            from nlptools.zoo.encoders.gru import GRUEncoder
            self.encoder = GRUEncoder(vocab_size=vocab_size,
                                      pretrained_embedding=pretrained_embedding,
                                      intermediate_size=encoder_intermediate_size,
                                      hidden_size=encoder_hidden_size,
                                      num_hidden_layers=encoder_hidden_layers,
                                      dropout=dropout)
            self.config = self.encoder.config
        elif bert_model_name is not None:
            from pytorch_pretrained_bert.modeling import BertModel
            self.encoder = BertModel.from_pretrained(bert_model_name)
            self.bert_model_name = bert_model_name
            self.config = self.encoder.config.to_dict()
        else:
            from nlptools.zoo.encoders.transformer import TransformerEncoder
            self.encoder = TransformerEncoder(vocab_size=vocab_size,
                                              pretrained_embedding=pretrained_embedding,
                                              num_hidden_layers=encoder_hidden_layers,
                                              num_attention_heads=encoder_attention_heads,
                                              max_position_embeddings=max_position_embeddings,
                                              hidden_size=encoder_hidden_size,
                                              intermediate_size=encoder_intermediate_size,
                                              dropout=dropout)
            self.config = self.encoder.config
        self.embedding = self.encoder.embeddings

    def freeze(self):
        """
            freeze encoder parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = False # freeze bert parameter

    def forward(self, sentence, attention_mask, output_all_encoded_layers=False):
        sequence_output, pooled_output = self.encoder(sentence, attention_mask=attention_mask, output_all_encoded_layers=False)
        return sequence_output, pooled_output

