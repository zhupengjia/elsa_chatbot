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
                 encoder_hidden_layers=12, encoder_attention_heads=12, max_position_embeddings=512,
                 encoder_intermediate_size=3072, encoder_hidden_size=768, dropout=0.1, **args):
        super().__init__()
        if model_type == "lstm":
            from nlptools.zoo.encoders.lstm import LSTMEncoder
            self.encoder = LSTMEncoder(vocab_size=vocab_size,
                                       hidden_size=encoder_hidden_size,
                                       num_hidden_layers=encoder_hidden_layers,
                                       intermediate_size=encoder_intermediate_size,
                                       dropout=dropout)
        elif bert_model_name is not None:
            from nlptools.zoo.encoders.transformer import TransformerEncoder
            self.encoder = TransformerEncoder.from_pretrained(bert_model_name)
            self.bert_model_name = bert_model_name
        else:
            from nlptools.zoo.encoders.transformer import TransformerEncoder
            self.encoder = TransformerEncoder(vocab_size=vocab_size,
                                              num_hidden_layers=encoder_hidden_layers,
                                              num_attention_heads=encoder_attention_heads,
                                              max_position_embeddings=max_position_embeddings,
                                              intermediate_size=encoder_intermediate_size,
                                              hidden_size=encoder_hidden_size,
                                              dropout=dropout)
        self.config = self.encoder.config
        self.embedding = self.encoder.embeddings
        self.hidden_size = self.config["hidden_size"]

    def freeze(self):
        """
            freeze encoder parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = False # freeze bert parameter

    def forward(self, sentence, attention_mask, output_all_encoded_layers=False):
        sequence_output, pooled_output = self.encoder(sentence, attention_mask=attention_mask, output_all_encoded_layers=False)
        return sequence_output, pooled_output

