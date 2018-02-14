#!/usr/bin/env python
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
from .sentence_encoder import Sentence_Encoder
from .model_base import Model_Base

class Dialog_Tracker(Model_Base):
    def __init__(self, cfg, vocab, Nresponses):
        super().__init__(cfg, vocab)
        self.Nresponses = Nresponses
        self.network()

    def network(self):
        self.encoder = Sentence_Encoder(self.cfg, self.vocab)
        self.encoder.network()
        self.conv = nn.Conv2d(in_channels = 1, \
                out_channels = self.cfg['cnn_kernel_num'], \
                kernel_size = (self.cfg['cnn_kernel_size'], self.vocab.emb_ins.vec_len),\
                padding = 0)
        self.dropout = nn.Dropout(self.cfg['dropout'])
        self.pool = nn.AvgPool1d(2)
        self.fc1 = nn.Linear(self.cfg['cnn_kernel_num']*2, self.cfg['cnn_kernel_num'])
        self.fc2 = nn.Linear(self.cfg['cnn_kernel_num']*3, self.Nresponses)
        self.softmax = nn.LogSoftmax()

    def entityencoder(self, x):
        x = self.encoder.embedding(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        x = self.pool(x)
        x = self.dropout(x)
        x_att = self.attention(x, x)
        return self.fc1(x_att)

    def forward(self, utterance, entity, mask):
        utterance = self.encoder(utterance)
        entity = self.entityencoder(entity)
        utter_att = self.attention(utterance, utterance)
        utter = torch.cat((utter_att, entity), 1) 
        response = self.fc2(utter)
        response = self.softmax(response)

        #response = response * mask

        return response

